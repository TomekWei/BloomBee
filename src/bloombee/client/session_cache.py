import threading
class SessionCache:
    def __init__(self):
        self._session = None
        self._lock = threading.Lock()
    
    def set_session(self, session):
        with self._lock:
            self._session = session
            print(f"SessionCache: set session {type(session)}")
    
    def get_session(self):
        with self._lock:
            return self._session
    
    def clear_session(self):
        with self._lock:
            self._session = None
            print("SessionCache: cleared session")


_session_cache = SessionCache()

def set_cached_session(session):
    _session_cache.set_session(session)

def get_cached_session():
    return _session_cache.get_session()

def clear_cached_session():
    _session_cache.clear_session()