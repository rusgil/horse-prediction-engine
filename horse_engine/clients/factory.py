"""Returns the appropriate racing data client."""


def get_tab_client():
    from horse_engine.clients.punters import PuntersClient
    return PuntersClient()
