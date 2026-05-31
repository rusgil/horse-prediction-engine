"""Returns the appropriate racing data client."""

_client = None


def get_tab_client():
    global _client
    if _client is None:
        from horse_engine.clients.composite import CompositeClient
        _client = CompositeClient()
    return _client
