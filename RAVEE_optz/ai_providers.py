"""Stage-specific provider selection. Imports/configuration never contact APIs."""

import gemini_runtime as gemini


def get_provider(stage):
    variable = {'label': 'RAVEE_LABEL_PROVIDER', 'aggregation': 'RAVEE_AGGREGATION_PROVIDER'}[stage]
    provider = (gemini.configuration().get(variable) or 'gemini').strip().lower()
    if provider not in ('gemini', 'twelvelabs'):
        raise RuntimeError(f'{variable} must be gemini or twelvelabs; no provider fallback performed')
    return provider


def get_model(stage):
    if get_provider(stage) == 'gemini':
        return gemini.get_model(stage)
    return (gemini.configuration().get('TWELVELABS_MODEL') or 'pegasus1.6').strip()


def provider_status(stage):
    try:
        return gemini.redact_text(f'{get_provider(stage)} / {get_model(stage)}', 140)
    except RuntimeError as error:
        return 'AI configuration error: ' + gemini.redact_text(error, 180)


def create_adapter(stage, **kwargs):
    if get_provider(stage) != 'twelvelabs':
        raise RuntimeError('Gemini uses the existing V3 request session')
    from twelvelabs_provider import TwelveLabsAdapter
    return TwelveLabsAdapter(model=get_model(stage), **kwargs)
