"""Shared, dependency-free visual system for the RAVEE desktop workspace."""

from tkinter import font, ttk


COLORS = {
    'background': '#F1F3EE',
    'surface': '#FBFCF9',
    'stripe': '#F1F5EF',
    'ink': '#253A34',
    'muted': '#596C63',
    'line': '#D4DED5',
    'accent': '#356451',
    'hover': '#294F40',
    'wash': '#E3ECE3',
    'disabled': '#718078',
    'preview': '#202C28',
}


def apply_theme(root):
    """Style one Tk interpreter; importing this module never creates a window."""
    c = COLORS
    root.configure(background=c['background'])
    for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont'):
        font.nametofont(name, root=root).configure(family='Segoe UI', size=10)
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', background=c['background'], foreground=c['ink'], font=('Segoe UI', 10))
    style.configure('TFrame', background=c['background'])
    style.configure('Card.TFrame', background=c['surface'])
    style.configure('TLabel', background=c['background'], foreground=c['ink'])
    style.configure('Card.TLabel', background=c['surface'])
    style.configure('Muted.TLabel', foreground=c['muted'])
    style.configure('CardMuted.TLabel', background=c['surface'], foreground=c['muted'])
    style.configure('Title.TLabel', font=('Segoe UI', 22, 'bold'))
    style.configure('Brand.TLabel', foreground=c['accent'], font=('Segoe UI', 10, 'bold'))
    style.configure('Section.TLabel', background=c['surface'], font=('Segoe UI', 11, 'bold'))
    style.configure('Badge.TLabel', background=c['wash'], foreground=c['accent'], padding=(12, 6))
    style.configure('TLabelframe', background=c['surface'], bordercolor=c['line'], relief='solid')
    style.configure('TLabelframe.Label', background=c['surface'], foreground=c['muted'])
    style.configure('TButton', background=c['surface'], foreground=c['ink'],
                    bordercolor=c['line'], lightcolor=c['surface'], darkcolor=c['surface'],
                    padding=(12, 9), focusthickness=2, focuscolor=c['accent'], borderwidth=1)
    style.map('TButton', background=[('disabled', c['background']), ('pressed', c['wash']),
                                    ('active', c['wash'])],
              foreground=[('disabled', c['disabled'])],
              bordercolor=[('focus', c['accent']), ('active', c['accent'])])
    style.configure('Primary.TButton', background=c['accent'], foreground='#FFFFFF',
                    bordercolor=c['accent'], lightcolor=c['accent'], darkcolor=c['accent'])
    style.map('Primary.TButton', background=[('disabled', c['line']), ('pressed', c['hover']),
                                            ('active', c['hover'])],
              foreground=[('disabled', c['muted']), ('!disabled', '#FFFFFF')],
              bordercolor=[('focus', c['ink'])])
    style.configure('TEntry', fieldbackground=c['surface'], padding=(10, 9),
                    bordercolor=c['line'], lightcolor=c['line'], darkcolor=c['line'],
                    insertcolor=c['ink'])
    style.map('TEntry', bordercolor=[('focus', c['accent'])])
    style.configure('TCombobox', padding=(8, 8), fieldbackground=c['surface'],
                    background=c['wash'], arrowcolor=c['accent'], bordercolor=c['line'])
    style.map('TCombobox', fieldbackground=[('readonly', c['surface'])],
              selectbackground=[('readonly', c['surface'])],
              selectforeground=[('readonly', c['ink'])], bordercolor=[('focus', c['accent'])])
    style.configure('Treeview', background=c['surface'], fieldbackground=c['surface'],
                    foreground=c['ink'], rowheight=34, borderwidth=0,
                    lightcolor=c['surface'], darkcolor=c['surface'])
    style.configure('Treeview.Heading', background=c['wash'], foreground=c['ink'],
                    font=('Segoe UI', 10, 'bold'), padding=(10, 10), relief='flat')
    style.map('Treeview', background=[('selected', c['accent'])],
              foreground=[('selected', '#FFFFFF')], bordercolor=[('focus', c['accent'])])
    style.map('Treeview.Heading', background=[('active', c['line'])])
    style.configure('Vertical.TScrollbar', background=c['line'], troughcolor=c['surface'],
                    arrowcolor=c['muted'], borderwidth=0, width=12)
    style.configure('Horizontal.TProgressbar', background=c['accent'],
                    troughcolor=c['line'], borderwidth=0, thickness=5)
    root.option_add('*TCombobox*Listbox.background', c['surface'])
    root.option_add('*TCombobox*Listbox.foreground', c['ink'])
    root.option_add('*TCombobox*Listbox.selectBackground', c['accent'])
    root.option_add('*TCombobox*Listbox.selectForeground', '#FFFFFF')
    return style


def workspace_header(root, title, subtitle, step):
    header = ttk.Frame(root, padding=(24, 20, 24, 16))
    header.grid(row=0, column=0, sticky='ew')
    header.columnconfigure(0, weight=1)
    ttk.Label(header, text='RAVEE   /   Video process discovery', style='Brand.TLabel').grid(
        row=0, column=0, sticky='w')
    ttk.Label(header, text=step, style='Badge.TLabel').grid(row=0, column=1, sticky='e')
    ttk.Label(header, text=title, style='Title.TLabel').grid(row=1, column=0, sticky='w', pady=(8, 3))
    ttk.Label(header, text=subtitle, style='Muted.TLabel').grid(row=2, column=0, columnspan=2, sticky='w')
    return header


def style_text(widget, *, readonly=False):
    widget.configure(background=COLORS['surface'], foreground=COLORS['ink'],
                     insertbackground=COLORS['ink'], selectbackground=COLORS['accent'],
                     selectforeground='white', font=('Segoe UI', 10), relief='flat',
                     borderwidth=0, padx=12, pady=10, highlightthickness=1,
                     highlightbackground=COLORS['line'], highlightcolor=COLORS['accent'])
    if readonly:
        widget.configure(cursor='arrow')


def fit_window(root, width, height, min_width, min_height):
    """Keep the initial workspace inside the desktop, including a small laptop."""
    width = min(width, root.winfo_screenwidth() - 64)
    height = min(height, root.winfo_screenheight() - 100)
    root.geometry(f'{width}x{height}')
    root.minsize(min(min_width, width), min(min_height, height))
