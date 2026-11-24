"""Keep catalog routing identifiers out of model prose and visible answers."""

import re

INTERNAL_ID = re.compile(r'WW-[0-9a-f]{32}', re.I)
PARTIAL_ID = re.compile(r'(?:W|WW|WW-[0-9a-f]{0,31})$', re.I)


def public_artifact_text(text, sources=(), *, language='zh'):
    names = {}
    for source in sources:
        metadata = source.get('metadata', {})
        identifier = metadata.get('artifact_id', '')
        for claim in metadata.get('claims', []):
            if claim['field'] == 'name' and claim.get('available', True):
                names[identifier.upper()] = claim['value']
                break
    fallback = 'the artifact' if language == 'en' else '该文物'
    return INTERNAL_ID.sub(lambda match: names.get(match[0].upper(), fallback)
                           if language != 'en' else fallback, text)


class PublicArtifactEvents:
    """Buffer only a possible ID suffix, including IDs split across SSE tokens."""

    def __init__(self, emit, sources=(), *, language='zh'):
        self.emit, self.sources, self.language = emit, sources, language
        self.pending = ''

    def __call__(self, event, data):
        if event == 'reset':
            self.pending = ''
        if event != 'delta':
            self.emit(event, data)
            return
        self.pending += data['text']
        partial = PARTIAL_ID.search(self.pending)
        end = partial.start() if partial else len(self.pending)
        visible, self.pending = self.pending[:end], self.pending[end:]
        self._emit(visible)

    def _emit(self, text):
        if text:
            self.emit('delta', {'text': public_artifact_text(text, self.sources, language=self.language)})

    def finish(self):
        self._emit(self.pending)
        self.pending = ''
