"""Offline GenerateContent fixtures; these are not captured live Hydrogen outputs."""

import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from google import genai
from google.genai import types

import label_suggester as labels
from tests import test_process_model_and_labels_v3 as fixtures


HYDROGEN = 'gemini-4-hydrogen'
OLD_MODEL = 'gemini-3.5-flash-lite'


def payload(**updates):
    value = dict(label='Rotate copper cylinder', evidence='Fingers visibly rotate the cylinder in the target.',
                 uncertainty='low', target_visible=True, bow_action='not_applicable')
    value.update(updates)
    return value


def sdk_response(value=None, *, text=None, finish='STOP', thoughts=False):
    parts = [types.Part(text='Private thought fixture, never log this', thought=True)] if thoughts else []
    if value is not None:
        text = json.dumps(value)
    if text is not None:
        parts.append(types.Part(text=text))
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role='model', parts=parts), finish_reason=finish)],
        model_version=HYDROGEN,
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=200, candidates_token_count=80, thoughts_token_count=432, total_token_count=712))


class ResponseValidationTests(unittest.TestCase):
    def reject(self, response, expected):
        with self.assertLogs(labels.__name__, level='ERROR') as logs:
            with self.assertRaises(labels.ActivityResponseError) as caught:
                labels._clean_label(response, model=HYDROGEN)
        self.assertIn(expected, str(caught.exception))
        self.assertTrue(logs.output)
        return caught.exception.diagnostics

    def test_valid_hydrogen_sdk_response_with_thought_parts(self):
        response = sdk_response(payload(), thoughts=True)
        self.assertEqual(labels._clean_label(response, model=HYDROGEN), 'Rotate copper cylinder')

    def test_all_required_fields_reported_individually_as_missing(self):
        for field in payload():
            with self.subTest(field=field):
                value = payload()
                del value[field]
                diagnostic = self.reject(sdk_response(value), f'missing fields: {field}')
                self.assertTrue(diagnostic['json_valid'])
                self.assertEqual(diagnostic['parsed_json'], value)

    def test_evidence_objects_lists_and_null_are_not_coerced_into_strings(self):
        for evidence in ({'motion': 'rotation'}, ['visible rotation'], None, 123, True):
            with self.subTest(evidence=evidence):
                self.reject(sdk_response(payload(evidence=evidence)), 'evidence must be a string')

    def test_strict_boolean_and_enums_are_preserved(self):
        for value, expected in ((dict(target_visible='true'), 'JSON boolean'),
                                (dict(target_visible=1), 'JSON boolean'),
                                (dict(uncertainty='certain'), 'uncertainty'),
                                (dict(bow_action='none'), 'bow_action')):
            with self.subTest(value=value):
                self.reject(sdk_response(payload(**value)), expected)

    def test_empty_responses_report_usage_and_finish_reason(self):
        for text in (None, '', '   '):
            with self.subTest(text=text):
                diagnostic = self.reject(sdk_response(text=text), 'empty or non-text')
                self.assertTrue(diagnostic['empty'])
                self.assertFalse(diagnostic['json_valid'])
                self.assertEqual(diagnostic['finish_reasons'], ['STOP'])
                self.assertEqual(diagnostic['usage_metadata']['thoughts_token_count'], 432)

    def test_max_tokens_with_empty_or_even_valid_json_is_not_accepted(self):
        for response in (sdk_response(text=None, finish='MAX_TOKENS', thoughts=True),
                         sdk_response(text='{"label":"Rotate', finish='MAX_TOKENS'),
                         sdk_response(payload(), finish='MAX_TOKENS')):
            with self.subTest(response=response.finish_reasons if hasattr(response, 'finish_reasons') else 'fixture'):
                diagnostic = self.reject(response, 'MAX_TOKENS')
                self.assertTrue(diagnostic['truncated'])
                self.assertNotIn('Private thought fixture', json.dumps(diagnostic))

    def test_invalid_json_reports_exact_parse_location(self):
        diagnostic = self.reject(sdk_response(text='{"label":"Rotate cylinder",'), 'JSON syntax:')
        self.assertIn('column', diagnostic['rule'])
        self.assertFalse(diagnostic['json_valid'])
        self.assertFalse(diagnostic['truncated'])  # server said STOP, not MAX_TOKENS

    def test_blocked_and_other_unsuccessful_candidates_are_actionable(self):
        response = types.GenerateContentResponse(prompt_feedback=types.GenerateContentResponsePromptFeedback(
            block_reason='SAFETY'))
        diagnostic = self.reject(response, 'blocked response (SAFETY)')
        self.assertTrue(diagnostic['blocked'])
        self.assertEqual(diagnostic['candidate_count'], 0)
        self.reject(sdk_response(payload(), finish='MALFORMED_FUNCTION_CALL'), 'MALFORMED_FUNCTION_CALL')

    def test_json_root_and_label_constraints_not_weakened(self):
        self.reject(sdk_response(text='[]'), 'JSON root must be an object')
        for label in ('', 'Rotate', 'one two three four five six seven', '!!! ???', 'A' * 101 + ' part'):
            with self.subTest(label=label):
                self.reject(sdk_response(payload(label=label)), 'label must be 2-to-6 words')
        # Compatibility metadata must not hide an invalid label before validation.
        self.reject(sdk_response(payload(label='', bow_action='tie_new')), 'label must be 2-to-6 words')

    def test_valid_insufficient_evidence_remains_unclear_without_inventing_evidence(self):
        for updates in (dict(uncertainty='high'), dict(target_visible=False), dict(evidence=''),
                        dict(bow_action='unclear')):
            self.assertEqual(labels._clean_label(sdk_response(payload(**updates)), model=HYDROGEN), 'Unclear activity')

    def test_sdk_parsed_field_does_not_rescue_malformed_or_missing_raw_json(self):
        response = sdk_response(text='{broken')
        response.parsed = payload()
        self.reject(response, 'JSON syntax:')
        response = sdk_response(text=None)
        response.parsed = payload()
        self.reject(response, 'empty or non-text')

    def test_legacy_text_response_without_candidate_metadata_still_works(self):
        response = SimpleNamespace(text=json.dumps(payload()))
        self.assertEqual(labels._clean_label(response, model=OLD_MODEL), 'Rotate copper cylinder')
        self.assertEqual(labels._clean_label(sdk_response(payload(bow_action='tighten_existing')), model=OLD_MODEL),
                         'Tighten ribbon bow')


class RequestCompatibilityTests(unittest.TestCase):
    def test_hydrogen_profile_uses_supported_schema_thinking_and_output_budget(self):
        for model in (HYDROGEN, 'models/' + HYDROGEN, HYDROGEN + '-001'):
            config = labels._generation_config(model, types)
            self.assertEqual(config.thinking_config.thinking_level, types.ThinkingLevel.LOW)
            self.assertIsNone(config.thinking_config.thinking_budget)
            self.assertIsNone(config.temperature)
            self.assertIsNone(config.top_p)
            self.assertIsNone(config.top_k)
            self.assertEqual(config.max_output_tokens, 65536)
            self.assertEqual(config.response_mime_type, 'application/json')
            self.assertIsNone(config.response_schema)  # only one schema mechanism
            schema = config.response_json_schema
            self.assertEqual(set(schema['required']), set(payload()))
            self.assertEqual(schema['properties']['evidence']['type'], 'string')
            self.assertEqual(schema['properties']['target_visible']['type'], 'boolean')

    def test_old_model_settings_are_unchanged_except_declaring_the_existing_schema(self):
        for model in (OLD_MODEL, 'gemini-3.8-flash'):
            config = labels._generation_config(model, types)
            self.assertEqual(config.temperature, .1)
            self.assertEqual(config.max_output_tokens, 512)
            self.assertIsNone(config.thinking_config)
            self.assertEqual(config.response_json_schema, labels._ACTIVITY_SCHEMA)

    def test_hydrogen_and_old_model_requests_keep_media_and_selected_model(self):
        for model in (HYDROGEN, OLD_MODEL):
            with self.subTest(model=model), patch.dict(os.environ, {
                'GEMINI_API_KEY': 'offline-secret', 'GOOGLE_API_KEY': '', 'GEMINI_MODEL': model,
                'RAVEE_SUGGEST_MODE': 'video', 'RAVEE_SUGGEST_CONTEXT_SECONDS': '3'}), \
                 patch.object(genai, 'Client') as factory, \
                 patch.object(labels.cv2, 'VideoCapture', side_effect=lambda _: fixtures.capture()), \
                 patch.object(labels, '_make_clip', return_value=b'synthetic MP4'):
                client = factory.return_value.__enter__.return_value
                client.models.generate_content.return_value = sdk_response(payload())
                self.assertEqual(labels.suggest_activity('synthetic.avi', 90, 149), 'Rotate copper cylinder')
                request = client.models.generate_content.call_args.kwargs
                self.assertEqual(request['model'], model)
                self.assertEqual(sum(part.inline_data is not None for part in request['contents'].parts), 3)
                self.assertTrue(all(part.inline_data.mime_type == 'video/mp4'
                                    for part in request['contents'].parts if part.inline_data is not None))
                client.models.generate_content.assert_called_once()

    def test_malformed_response_has_no_automatic_retry_or_fabricated_success(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'offline-secret', 'GOOGLE_API_KEY': '', 'GEMINI_MODEL': HYDROGEN}), \
             patch.object(genai, 'Client') as factory, \
             patch.object(labels, '_target_context_windows', return_value=(15, [('TARGET SEGMENT', 0, 29)])), \
             patch.object(labels, '_make_clip', return_value=b'synthetic MP4'), \
             self.assertLogs(labels.__name__, level='ERROR'):
            client = factory.return_value.__enter__.return_value
            client.models.generate_content.return_value = sdk_response(payload(evidence={}))
            with self.assertRaisesRegex(labels.ActivityResponseError, 'evidence must be a string'):
                labels.suggest_activity('synthetic.avi', 0, 29)
            client.models.generate_content.assert_called_once()


class DiagnosticPrivacyTests(unittest.TestCase):
    def test_invalid_raw_and_parsed_json_redact_keys_credentials_and_bearer_tokens(self):
        key = 'test-active-api-key-secret'
        value = payload(evidence={'api_key': key, 'password': 'sensitive password with spaces',
                                  'authorization': 'Bearer secret-access-token',
                                  'private_key': 'sensitive fake private value'}, label='Rotate cylinder')
        with patch.dict(os.environ, {'GEMINI_API_KEY': key}), self.assertLogs(labels.__name__, level='ERROR') as logs:
            with self.assertRaises(labels.ActivityResponseError) as caught:
                labels._clean_label(sdk_response(value), model=HYDROGEN)
        output = '\n'.join(logs.output) + json.dumps(caught.exception.diagnostics)
        for secret in (key, 'sensitive password with spaces', 'secret-access-token', 'sensitive fake private value'):
            self.assertNotIn(secret, output)
        self.assertIn('evidence must be a string', output)
        self.assertIn('REDACTED', output)

    def test_inactive_or_prefixed_credentials_are_redacted_without_reading_environment(self):
        for key in ('GEMINI_API_KEY', 'X-Goog-Api-Key', 'token', 'private_key', 'service_credentials'):
            with self.subTest(key=key):
                text = json.dumps({key: 'unknown credential with spaces'})
                self.assertNotIn('unknown credential', labels._redact_text(text))

    def test_api_error_details_are_distinguished_and_redacted(self):
        class ApiError(RuntimeError):
            code = 400
            status = 'INVALID_ARGUMENT'
            details = {'api_key': 'do-not-log-this-key'}
        error = ApiError('Invalid schema; api_key=do-not-log-this-key')
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'do-not-log-this-key', 'GOOGLE_API_KEY': '', 'GEMINI_MODEL': HYDROGEN}), \
             patch.object(genai, 'Client', side_effect=error), \
             patch.object(labels, '_target_context_windows', return_value=(15, [('TARGET SEGMENT', 0, 29)])), \
             patch.object(labels, '_make_clip', return_value=b'synthetic MP4'), \
             self.assertLogs(labels.__name__, level='ERROR') as logs:
            with self.assertRaisesRegex(RuntimeError, 'API request failed') as caught:
                labels.suggest_activity('synthetic.avi', 0, 29)
        output = '\n'.join(logs.output)
        self.assertNotIn('do-not-log-this-key', output + str(caught.exception))
        self.assertIn('400', output)
        self.assertIn('INVALID_ARGUMENT', output)
        self.assertIn('activity_api_error', output)
        self.assertTrue(caught.exception.__suppress_context__)

    def test_diagnostic_size_is_bounded_and_no_raw_response_object_is_dumped(self):
        response = sdk_response(payload(evidence={'unexpected': 'X' * 50000}))
        response.sdk_http_response = types.HttpResponse(headers={'x-goog-api-key': 'never-dump-headers'})
        with self.assertLogs(labels.__name__, level='ERROR') as logs:
            with self.assertRaises(labels.ActivityResponseError):
                labels._clean_label(response, model=HYDROGEN)
        output = '\n'.join(logs.output)
        self.assertNotIn('never-dump-headers', output)
        self.assertIn('diagnostic clipped', output)
        self.assertLess(len(output), 19000)


if __name__ == '__main__':
    unittest.main()
