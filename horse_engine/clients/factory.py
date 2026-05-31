"""Returns the appropriate racing data client."""

_client = None


def get_tab_client():
    global _client
    if _client is None:
        from horse_engine.clients.tab import TABClient
        _client = TABClient()
    return _client
