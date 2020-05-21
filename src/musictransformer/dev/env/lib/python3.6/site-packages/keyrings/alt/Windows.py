import sys
import base64
import platform
import functools

from keyring.util import properties
from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError, ExceptionRaisedContext
from . import file_base

try:
    # prefer pywin32-ctypes
    __import__('win32ctypes.pywintypes')
    from win32ctypes import win32cred
    # force demand import to raise ImportError
    win32cred.__name__
except ImportError:
    # fallback to pywin32
    try:
        __import__('pywintypes')
        import win32cred
    except ImportError:
        pass

try:
    import winreg
except ImportError:
    try:
        # Python 2 compatibility
        import _winreg as winreg
    except ImportError:
        pass

try:
    from . import _win_crypto
except ImportError:
    pass


def has_pywin32():
    """
    Does this environment have pywin32?
    Should return False even when Mercurial's Demand Import allowed import of
    win32cred.
    """
    with ExceptionRaisedContext() as exc:
        win32cred.__name__
    return not bool(exc)


def has_wincrypto():
    """
    Does this environment have wincrypto?
    Should return False even when Mercurial's Demand Import allowed import of
    _win_crypto, so accesses an attribute of the module.
    """
    with ExceptionRaisedContext() as exc:
        _win_crypto.__name__
    return not bool(exc)


class EncryptedKeyring(file_base.Keyring):
    """
    A File-based keyring secured by Windows Crypto API.
    """

    @properties.ClassProperty
    @classmethod
    def priority(self):
        """
        Preferred over file.EncryptedKeyring but not other, more sophisticated
        Windows backends.
        """
        if not platform.system() == 'Windows':
            raise RuntimeError("Requires Windows")
        return .8

    filename = 'wincrypto_pass.cfg'

    def encrypt(self, password):
        """Encrypt the password using the CryptAPI.
        """
        return _win_crypto.encrypt(password)

    def decrypt(self, password_encrypted):
        """Decrypt the password using the CryptAPI.
        """
        return _win_crypto.decrypt(password_encrypted)


class RegistryKeyring(KeyringBackend):
    """
    RegistryKeyring is a keyring which use Windows CryptAPI to encrypt
    the user's passwords and store them under registry keys
    """

    @properties.ClassProperty
    @classmethod
    def priority(self):
        """
        Preferred on Windows when pywin32 isn't installed
        """
        if platform.system() != 'Windows':
            raise RuntimeError("Requires Windows")
        if not has_wincrypto():
            raise RuntimeError("Requires ctypes")
        return 2

    def get_password(self, service, username):
        """Get password of the username for the service
        """
        try:
            # fetch the password
            key = r'Software\%s\Keyring' % service
            hkey = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key)
            password_saved = winreg.QueryValueEx(hkey, username)[0]
            password_base64 = password_saved.encode('ascii')
            # decode with base64
            password_encrypted = base64.decodestring(password_base64)
            # decrypted the password
            password = _win_crypto.decrypt(password_encrypted).decode('utf-8')
        except EnvironmentError:
            password = None
        return password

    def set_password(self, service, username, password):
        """Write the password to the registry
        """
        # encrypt the password
        password_encrypted = _win_crypto.encrypt(password.encode('utf-8'))
        # encode with base64
        password_base64 = base64.encodestring(password_encrypted)
        # encode again to unicode
        password_saved = password_base64.decode('ascii')

        # store the password
        key_name = r'Software\%s\Keyring' % service
        hkey = winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_name)
        winreg.SetValueEx(hkey, username, 0, winreg.REG_SZ, password_saved)

    def delete_password(self, service, username):
        """Delete the password for the username of the service.
        """
        try:
            key_name = r'Software\%s\Keyring' % service
            hkey = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, key_name, 0,
                winreg.KEY_ALL_ACCESS)
            winreg.DeleteValue(hkey, username)
            winreg.CloseKey(hkey)
        except WindowsError:
            e = sys.exc_info()[1]
            raise PasswordDeleteError(e)
        self._delete_key_if_empty(service)

    def _delete_key_if_empty(self, service):
        key_name = r'Software\%s\Keyring' % service
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, key_name, 0,
            winreg.KEY_ALL_ACCESS)
        try:
            winreg.EnumValue(key, 0)
            return
        except WindowsError:
            pass
        winreg.CloseKey(key)

        # it's empty; delete everything
        while key_name != 'Software':
            parent, sep, base = key_name.rpartition('\\')
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, parent, 0,
                winreg.KEY_ALL_ACCESS)
            winreg.DeleteKey(key, base)
            winreg.CloseKey(key)
            key_name = parent


class OldPywinError(object):
    """
    A compatibility wrapper for old PyWin32 errors, such as reported in
    https://bitbucket.org/kang/python-keyring-lib/issue/140/
    """
    def __init__(self, orig):
        self.orig = orig

    @property
    def funcname(self):
        return self.orig[1]

    @property
    def winerror(self):
        return self.orig[0]

    @classmethod
    def wrap(cls, orig_err):
        attr_check = functools.partial(hasattr, orig_err)
        is_old = not all(map(attr_check, ['funcname', 'winerror']))
        return cls(orig_err) if is_old else orig_err
