"""Synthetic/offline checks for preview sizing, segment scope and reviewed labels."""

import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from google import genai

import label_suggester as labels
from gui import gui_elements, semantic_review
from gui.video_preview import fit_preview_size, resize_preview_image
from toolbox import main as app
from toolbox import event_log_creation as events
from toolbox import semantic_aggregation as semantic
from tests import test_process_model_and_labels_v3 as v3_fixtures
from tests import test_semantic_aggregation as semantic_fixtures


ROOT = Path(__file__).resolve().parents[1]


class PreviewSizeTests(unittest.TestCase):
    def test_small_source_upscales_to_400_pixel_preview_height(self):
        self.assertEqual(fit_preview_size(320, 240, 760, 400), (533, 400))
        source = Image.new('RGB', (320, 240), '#426955')
        result = resize_preview_image(source, 760, 400)
        self.assertEqual(result.size, (533, 400))
        self.assertEqual(source.size, (320, 240))

    def test_landscape_portrait_and_square_preserve_ratio_and_use_limiting_edge(self):
        for source in ((1920, 1080), (640, 480), (240, 320), (128, 128)):
            for box in ((800, 450), (700, 400), (420, 350), (300, 180)):
                with self.subTest(source=source, box=box):
                    width, height = fit_preview_size(*source, *box)
                    self.assertLessEqual(width, box[0])
                    self.assertLessEqual(height, box[1])
                    self.assertTrue(abs(width - box[0]) <= 1 or abs(height - box[1]) <= 1)
                    self.assertAlmostEqual(width / height, source[0] / source[1], delta=.012)

    def test_resizing_grows_and_shrinks_rendered_image_without_native_size_ceiling(self):
        source = Image.new('RGB', (160, 90))
        self.assertEqual(resize_preview_image(source, 800, 450).size, (800, 450))
        self.assertEqual(resize_preview_image(source, 320, 180).size, (320, 180))
        self.assertEqual(resize_preview_image(source, 960, 540).size, (960, 540))

    def test_unmapped_widget_bounds_are_safe_and_bad_source_dimensions_rejected(self):
        self.assertEqual(fit_preview_size(640, 480, 0, 0), (1, 1))
        for source in ((0, 20), (20, 0), (-1, 20)):
            with self.assertRaises(ValueError):
                fit_preview_size(*source, 800, 450)


class SemanticPreviewWorkflowTests(unittest.TestCase):
    def test_resize_playback_and_all_manual_review_controls_without_tk_desktop(self):
        widgets, timers = [], {}

        def widget(kind, *args, **options):
            item = Mock()
            item.kind, item.options, item.bindings = kind, options.copy(), {}
            item.initial_options = options.copy()
            item.value, item.children, item.chosen = '', [], ()
            item.winfo_width.return_value, item.winfo_height.return_value = 800, 450
            item.configure.side_effect = lambda **kwargs: item.options.update(kwargs)
            item.bind.side_effect = lambda event, callback: item.bindings.update({event: callback})
            item.get.side_effect = lambda *args: item.value
            item.set.side_effect = lambda value: setattr(item, 'value', value)
            item.get_children.side_effect = lambda: tuple(item.children)
            item.selection.side_effect = lambda: item.chosen
            item.selection_set.side_effect = lambda values: setattr(item, 'chosen',
                (values,) if isinstance(values, str) else tuple(values))
            def insert(*args, **kwargs):
                if kind == 'Treeview':
                    item.children.append(kwargs['iid'])
                else:
                    item.value = args[1]
            def delete(*args):
                item.value = ''
                if kind == 'Treeview':
                    item.children.clear()
                    item.chosen = ()
            item.insert.side_effect, item.delete.side_effect = insert, delete
            widgets.append(item)
            return item

        root = widget('Root')
        serial = 0
        def after(delay, callback):
            nonlocal serial
            serial += 1
            key = f'timer-{serial}'
            timers[key] = (delay, callback)
            return key
        root.after.side_effect = after
        root.after_cancel.side_effect = lambda key: timers.pop(key, None)
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: 15 if prop == semantic_review.cv2.CAP_PROP_FPS else 0
        cap.read.return_value = (True, frame)
        detailed = semantic_fixtures.detailed(['Rotate cylinder', 'Inspect cylinder'])

        def find(kind, text=None):
            return next(item for item in widgets if item.kind == kind
                        and (text is None or item.initial_options.get('text') == text))

        def exercise():
            preview = find('Label', 'Select a task to preview its video')
            self.assertEqual(preview.options['image'].size, (800, 450))
            player = find('Button', 'Play selected task')
            player.options['command']()
            reads = cap.read.call_count
            preview.winfo_width.return_value, preview.winfo_height.return_value = 300, 200
            preview.bindings['<Configure>'](None)
            preview.bindings['<Configure>'](None)
            scheduled = [(key, callback) for key, (delay, callback) in timers.items() if delay == 40]
            self.assertEqual(len(scheduled), 1)
            key, redraw = scheduled[0]
            timers.pop(key)
            redraw()
            self.assertEqual(preview.options['image'].size, (300, 169))
            self.assertEqual(cap.read.call_count, reads)  # resize does not seek/read/restart
            player.options['command']()  # stop existing playback

            listing = find('Treeview')
            listing.selection_set(('0', '1'))
            find('Button', 'Merge adjacent groups').options['command']()
            self.assertEqual(len(listing.children), 1)
            find('Button', 'Split group').options['command']()
            self.assertEqual(len(listing.children), 2)
            find('Button', 'Undo').options['command']()
            self.assertEqual(len(listing.children), 1)
            find('Button', 'Reset groups').options['command']()
            self.assertEqual(len(listing.children), 2)

            find('Entry').value = 'Rotate copper cylinder'
            reason = [item for item in widgets if item.kind == 'Text'][-1]
            reason.value = 'Visible rotation of this cylinder in the target.'
            find('Button', 'Apply edits').options['command']()
            self.assertEqual(find('Entry').value, 'Rotate copper cylinder')
            self.assertEqual(reason.value, 'Visible rotation of this cylinder in the target.')
            refresh = find('Button', 'Refresh AI suggestions')
            self.assertEqual(refresh.options['state'], 'disabled')
            refresh.options['command']()  # offline guard must still prevent a request
            self.assertTrue(callable(find('Button', 'Cancel review').options['command']))
            find('Button', 'Confirm task groups').options['command']()

        root.mainloop.side_effect = exercise
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(semantic_review.tk, 'Tk', return_value=root))
            for kind in ('PanedWindow', 'Label', 'Text', 'StringVar'):
                stack.enter_context(patch.object(semantic_review.tk, kind,
                    side_effect=lambda *args, _kind=kind, **kwargs: widget(_kind, *args, **kwargs)))
            for kind in ('Frame', 'Label', 'Button', 'Entry', 'Combobox', 'Treeview', 'Scrollbar'):
                stack.enter_context(patch.object(semantic_review.ttk, kind,
                    side_effect=lambda *args, _kind=kind, **kwargs: widget(_kind, *args, **kwargs)))
            for name in ('apply_theme', 'fit_window', 'workspace_header', 'style_text'):
                stack.enter_context(patch.object(semantic_review, name))
            stack.enter_context(patch.object(semantic_review.cv2, 'VideoCapture', return_value=cap))
            stack.enter_context(patch.object(semantic_review.ImageTk, 'PhotoImage',
                side_effect=lambda image: SimpleNamespace(size=image.size)))
            stack.enter_context(patch.object(semantic_review.messagebox, 'askyesno', return_value=True))
            warning = stack.enter_context(patch.object(semantic_review.messagebox, 'showwarning'))
            client = stack.enter_context(patch.object(genai, 'Client'))
            groups = semantic_review.review_task_groups('synthetic.avi', detailed, offline=True)
        self.assertEqual(groups[0]['suggested_task_label'], 'Rotate copper cylinder')
        self.assertEqual(groups[0]['explanation'], 'Visible rotation of this cylinder in the target.')
        self.assertEqual(detailed['activity'].tolist(), ['Rotate cylinder', 'Inspect cylinder'])
        self.assertEqual(find('PanedWindow').options['orient'], semantic_review.tk.HORIZONTAL)
        self.assertEqual(find('PanedWindow').add.call_count, 2)
        warning.assert_not_called()
        client.assert_not_called()
        cap.release.assert_called_once()
        self.assertFalse(timers)


class SegmentScopeTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'GEMINI_API_KEY': 'offline-only', 'GOOGLE_API_KEY': '',
            'RAVEE_SUGGEST_MODE': 'video', 'RAVEE_SUGGEST_CONTEXT_SECONDS': '3'})
        env.start()
        self.addCleanup(env.stop)

    def test_prompt_distinguishes_actions_from_overall_tasks_without_domain_examples(self):
        prompt = labels._PROMPT
        self.assertIn('specific observable action', prompt)
        self.assertIn('not the overall task, process', prompt)
        self.assertIn('narrowest verb + object', prompt)
        self.assertIn('evidence', prompt.lower())
        self.assertIn('Use exactly Unclear activity', prompt)
        for reference in ('Wrap gift', 'Attach decoration', 'pine sprig', 'Trim ribbon', 'Segments 1'):
            self.assertNotIn(reference.lower(), prompt.lower())

    def test_short_target_bounds_context_but_does_not_change_aggregation_context(self):
        with patch.object(labels.cv2, 'VideoCapture', side_effect=lambda _: v3_fixtures.capture()):
            _, original = labels._context_windows('synthetic.avi', 90, 104)
            _, bounded = labels._target_context_windows('synthetic.avi', 90, 104)
        self.assertEqual(original, [('BEFORE CONTEXT', 45, 89), ('TARGET SEGMENT', 90, 104),
                                    ('AFTER CONTEXT', 105, 149)])
        self.assertEqual(bounded, [('BEFORE CONTEXT', 75, 89), ('TARGET SEGMENT', 90, 104),
                                   ('AFTER CONTEXT', 105, 119)])
        self.assertLessEqual(sum(end-start+1 for role, start, end in bounded if role != 'TARGET SEGMENT'), 30)

    def test_context_still_respects_zero_setting_and_video_edges(self):
        for start, end in ((0, 5), (1530, 1538)):
            with patch.object(labels.cv2, 'VideoCapture', side_effect=lambda _: v3_fixtures.capture()):
                _, windows = labels._target_context_windows('synthetic.avi', start, end)
            for role, a, b in windows:
                self.assertGreaterEqual(a, 0)
                self.assertLess(b, 1539)
                if role != 'TARGET SEGMENT':
                    self.assertLessEqual(b-a+1, end-start+1)
        with patch.dict(os.environ, {'RAVEE_SUGGEST_CONTEXT_SECONDS': '0'}), \
             patch.object(labels.cv2, 'VideoCapture', return_value=v3_fixtures.capture()):
            self.assertEqual(labels._target_context_windows('synthetic.avi', 90, 104)[1],
                             [('TARGET SEGMENT', 90, 104)])

    def test_request_reanchors_exact_target_after_context_with_existing_json_contract(self):
        with patch.object(genai, 'Client') as factory, \
             patch.object(labels.cv2, 'VideoCapture', side_effect=lambda _: v3_fixtures.capture()), \
             patch.object(labels, '_make_clip', return_value=b'synthetic media') as clips:
            client = factory.return_value.__enter__.return_value
            client.models.generate_content.return_value = v3_fixtures.response(
                label='Rotate copper cylinder', evidence='The fingers visibly rotate the cylinder in the target.')
            self.assertEqual(labels.suggest_activity('synthetic.avi', 90, 104), 'Rotate copper cylinder')
        self.assertEqual([call.args[1:] for call in clips.call_args_list], [(75, 89), (90, 104), (105, 119)])
        request = client.models.generate_content.call_args.kwargs
        self.assertIn('ONLY TARGET SEGMENT source frames 90-104', request['contents'].parts[-1].text)
        self.assertIn('actual supplied target coverage is 90-104', request['contents'].parts[-1].text)
        self.assertEqual(request['config'].response_mime_type, 'application/json')
        self.assertEqual(request['config'].max_output_tokens, 512)
        self.assertEqual(request['model'], os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite'))
        self.assertEqual(sum(part.inline_data is not None for part in request['contents'].parts), 3)
        client.models.generate_content.assert_called_once()

    def test_structural_parser_preserves_contract_without_claiming_semantic_verification(self):
        result = v3_fixtures.response(label='Rotate copper cylinder')
        fields = json.loads(result.text)
        self.assertEqual(set(fields), {'label', 'evidence', 'uncertainty', 'target_visible', 'bow_action'})
        self.assertEqual(labels._clean_label(result), 'Rotate copper cylinder')
        self.assertEqual(labels._clean_label(v3_fixtures.response(uncertainty='high')), 'Unclear activity')
        # There is deliberately no domain keyword blacklist or made-up semantic
        # verifier: a confident but broad model answer still needs human review.
        self.assertEqual(labels._clean_label(v3_fixtures.response(label='Assemble complete machine')),
                         'Assemble complete machine')


class ReviewedLabelTests(unittest.TestCase):
    def test_actual_finish_callback_redirects_to_earlier_unsaved_edit(self):
        tree = ast.parse((ROOT / 'gui/gui_elements.py').read_text(encoding='utf-8'))
        labeling = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'label_video_clusters')
        callback = next(node for node in labeling.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'commit_and_next')
        namespace = dict(vars(gui_elements), LABEL={1: 'Broad process label', 2: 'Inspect part'},
            drafts={0: 'Rotate copper cylinder', 1: 'Inspect part'}, current_index=1,
            segments=[{}, {}], label_entry=Mock(), root=Mock(), messagebox=Mock(),
            update_progress_and_tree=Mock(), go_to_segment=Mock())
        namespace['label_entry'].get.return_value = 'Inspect part'
        exec(compile(ast.Module(body=[callback], type_ignores=[]), '<actual GUI callback>', 'exec'), namespace)
        namespace['commit_and_next']()
        namespace['go_to_segment'].assert_called_once_with(0)
        namespace['root'].destroy.assert_not_called()
        self.assertEqual(namespace['LABEL'][1], 'Broad process label')
        namespace['LABEL'][1] = namespace['drafts'][0]  # explicit earlier Commit
        namespace['commit_and_next']()
        namespace['root'].destroy.assert_called_once()

    def test_earlier_edited_draft_is_still_pending_when_last_segment_is_saved(self):
        committed = {1: 'Broad old activity', 2: 'Inspect part', 3: 'Put down tool'}
        drafts = {0: 'Rotate copper cylinder', 1: 'Inspect part', 2: 'Put down tool'}
        self.assertEqual(gui_elements.uncommitted_segments(committed, drafts, 3), [0])
        self.assertEqual(committed[1], 'Broad old activity')  # checking cannot auto-commit
        committed[1] = drafts[0]
        self.assertEqual(gui_elements.uncommitted_segments(committed, drafts, 3), [])

    def test_missing_empty_and_reverted_drafts(self):
        self.assertEqual(gui_elements.uncommitted_segments({1: 'Move tool'}, {0: 'Move tool'}, 2), [1])
        self.assertEqual(gui_elements.uncommitted_segments({1: 'Move tool'}, {0: ''}, 1), [0])
        self.assertEqual(gui_elements.uncommitted_segments({1: 'Move tool'}, {0: 'Move tool'}, 1), [])

    def test_real_main_and_extractlog_pass_final_reviewed_activity_to_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            case = dict(video_path='synthetic.avi', case_id=1, video_name='synthetic')
            clusters = np.array([[0], [0], [1], [1]])
            prepared = (object(), {'1-synthetic': case}, {'1-synthetic': (clusters, [2], None)})
            gui = SimpleNamespace(LABEL={}, counter=1)
            def review_labels(*args):
                # Cached raw suggestion is not the value committed by the human.
                gui.LABEL.update({1: 'Rotate copper cylinder', 2: 'Inspect cylinder surface'})
                return 0
            def aggregate(video, detailed, *args):
                self.assertEqual(detailed['activity'].tolist(), ['Rotate copper cylinder', 'Inspect cylinder surface'])
                self.assertEqual([row['activity'] for row in semantic.segment_records(detailed)],
                                 detailed['activity'].tolist())
                return detailed.copy()
            with patch.object(app, 'gui_elements', gui), \
                 patch.object(app, 'finchOutput', directory), patch.object(app, 'eventlogOutput', directory), \
                 patch.object(app, 'run_snippets', side_effect=review_labels), \
                 patch.object(app, 'extractLog', wraps=events.extractLog) as extract, \
                 patch.object(app, 'review_semantic_task_log', side_effect=aggregate) as aggregation, \
                 patch.object(app, 'export_two_event_logs'), patch.object(app, 'showProcessModel'), \
                 contextlib.redirect_stdout(io.StringIO()):
                app.main(videos=['synthetic.mp4'], prepared=prepared)
            extract.assert_called_once()
            aggregation.assert_called_once()
            self.assertEqual(gui.LABEL, {})  # main resets GUI after taking its copy

    def test_old_ai_provenance_cannot_replace_the_reviewed_activity_field(self):
        detailed = semantic_fixtures.detailed(['Rotate copper cylinder', 'Inspect cylinder surface'])
        detailed['ai_label'] = ['Assemble machine', 'Assemble machine']
        records = semantic.segment_records(detailed)
        self.assertEqual(records[0]['activity'], 'Rotate copper cylinder')
        self.assertEqual(records[0]['ai_label'], 'Assemble machine')


class ReviewedCacheTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.video = self.root / 'synthetic.avi'
        self.video.write_bytes(b'identity only; never decoded')
        self.df = semantic_fixtures.detailed(['Rotate copper cylinder', 'Inspect cylinder surface'])
        env = patch.dict(os.environ, {'GEMINI_API_KEY': 'offline-only', 'GOOGLE_API_KEY': ''})
        env.start()
        self.addCleanup(env.stop)

    def test_boundary_cache_reuses_unchanged_labels_but_never_changed_ones_or_fallbacks(self):
        def respond(**kwargs):
            records = json.loads(kwargs['contents'].parts[1].text)['segments']
            return SimpleNamespace(text=json.dumps(semantic_fixtures.proposal(records)))
        with patch.object(genai, 'Client') as factory, patch.object(semantic, '_media_parts', return_value=[]):
            client = factory.return_value.__enter__.return_value
            client.models.generate_content.side_effect = respond
            stats = {}
            semantic.propose_task_groups(self.video, self.df, cache_dir=self.root / 'cache', stats=stats)
            old = {path: path.read_bytes() for path in (self.root / 'cache').glob('*.json')}
            factory.reset_mock()
            semantic.propose_task_groups(self.video, self.df, cache_dir=self.root / 'cache', stats=stats)
            factory.assert_not_called()
            self.assertEqual(stats['cache_hits'], 1)
            self.assertEqual(stats['api_calls'], 0)
            before = semantic._boundary_cache_key(semantic._video_identity(self.video), semantic.segment_records(self.df), 0, 12)
            self.df.loc[0, 'activity'] = 'Place copper cylinder'
            after = semantic._boundary_cache_key(semantic._video_identity(self.video), semantic.segment_records(self.df), 0, 12)
            self.assertNotEqual(before, after)
            factory.assert_not_called()  # edit / key calculation cannot initiate AI
            client.models.generate_content.side_effect = RuntimeError('offline')
            semantic.propose_task_groups(self.video, self.df, cache_dir=self.root / 'cache', stats=stats,
                                         refresh_suggestions=True)
            self.assertEqual(stats['cache_hits'], 0)
            self.assertEqual(stats['fallback'], 0)
            self.assertEqual(stats['unresolved'], 1)
            self.assertTrue(all(path.read_bytes() == content for path, content in old.items()))

    def test_confirmed_review_changes_identity_on_label_edit_without_touching_old_file(self):
        records = semantic.segment_records(self.df)
        identity = semantic._review_identity(self.video, self.df)
        path = self.root / 'confirmed.json'
        semantic.save_confirmed_review(path, identity, semantic.singleton_groups(records))
        original = path.read_bytes()
        self.assertIsNotNone(semantic.load_confirmed_review(path, identity, records))
        with patch.object(genai, 'Client') as factory:
            self.df.loc[0, 'activity'] = 'Place copper cylinder'
            changed = semantic._review_identity(self.video, self.df)
            self.assertNotEqual(identity, changed)
            self.assertIsNone(semantic.load_confirmed_review(path, changed, semantic.segment_records(self.df)))
            factory.assert_not_called()
        self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
