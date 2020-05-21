import time
import datetime as datetime_module
import calendar


__author__ = 'jbenet@cs.stanford.edu'
__version__ = '0.5.2'


__doc__ = '''
The nanotime module provides a time object that keeps time as the number of
nanoseconds since the UNIX epoch. In other words, it is a UNIX timestamp with
nanosecond precision.

It's base type is a 64bit integer (for network use).
'''

class nanotime(object):

  def __init__(self, nanoseconds):
    if isinstance(nanoseconds, self.__class__):
      self._ns = nanoseconds._ns
    elif isinstance(nanoseconds, int):
      self._ns = nanoseconds
    else:
      self._ns = int(round(nanoseconds))

  #----------------------------------------------------
  def days(self):
    return self._ns / (1.0e9 * 60 * 60 * 24)

  def hours(self):
    return self._ns / (1.0e9 * 60 * 60)

  def minutes(self):
    return self._ns / (1.0e9 * 60)

  def seconds(self):
    return self._ns / 1.0e9

  def milliseconds(self):
    return self._ns / 1.0e6

  def microseconds(self):
    return self._ns / 1.0e3

  def nanoseconds(self):
    return self._ns

  #----------------------------------------------------

  def unixtime(self):
    return self.timestamp()

  def timestamp(self):
    if hasattr(self, '_timestamp'):
      return self._timestamp
    return self.seconds()

  def datetime(self):
    if hasattr(self, '_datetime'):
      return self._datetime
    return datetime_module.datetime.fromtimestamp(self.timestamp())

  #----------------------------------------------------

  def __int__(self):
    return self._ns

  def __str__(self):
    frac = str(self._ns)[-9:]
    # when microseconds == 000000, datetime doesnt print them.
    if frac[:6] == '000000':
      return '%s.%s' % (self.datetime(), frac)
    return '%s%s' % (self.datetime(), frac[-3:])

  def __repr__(self):
    return 'nanotime.nanotime(%d)' % self._ns

  #----------------------------------------------------

  def __add__(self, other):
    return self.__class__(self._ns + other._ns)

  def __sub__(self, other):
    return self.__class__(self._ns - other._ns)

  def __mul__(self, other):
    return self.__class__(self._ns * other._ns)

  def __div__(self, other):
    return self.__class__(self._ns * 1.0 / other._ns)

  def __cmp__(self, other):
    '''Attempt to convert other types'''
    if isinstance(other, self.__class__):
      return cmp(self._ns, other._ns)
    else:
      return cmp(self._ns, _converter.convert(other)._ns)

  def __hash__(self):
    return hash(self._ns)

  @classmethod
  def now(cls):
    return _converter.now()


class _converter(object):

  @classmethod
  def days(cls, m):
    return nanotime(m * 1000000000 * 60 * 60 * 24)

  @classmethod
  def hours(cls, m):
    return nanotime(m * 1000000000 * 60 * 60)

  @classmethod
  def minutes(cls, m):
    return nanotime(m * 1000000000 * 60)

  @classmethod
  def seconds(cls, s):
    return nanotime(s * 1000000000)

  @classmethod
  def milliseconds(cls, ms):
    return nanotime(ms * 1000000)

  @classmethod
  def microseconds(cls, us):
    return nanotime(us * 1000)

  @classmethod
  def nanoseconds(cls, ns):
    return nanotime(ns)

  #----------------------------------------------------

  @classmethod
  def unixtime(cls, unixtime):
    return cls.timestamp(unixtime)

  @classmethod
  def timestamp(cls, ts):
    nt = cls.seconds(ts)
    nt._timestamp = ts
    return nt

  @classmethod
  def datetime(cls, d):
    du = d if d.utcoffset() is None else d - d.utcoffset()
    us = int(calendar.timegm(du.timetuple()) * 1000000 + du.microsecond)
    nt = cls.microseconds(us)
    nt._datetime = d
    return nt

  #----------------------------------------------------

  @classmethod
  def now(cls):
    return cls.seconds(time.time())

  @classmethod
  def convert(cls, other):
    if isinstance(other, datetime_module.datetime):
      return cls.datetime(other)
    elif isinstance(other, float):
      return cls.timestamp(other)
    elif isinstance(other, int) or isinstance(other, long):
      return cls.nanoseconds(other)
    else:
      raise TypeError('Cannot convert %s into %s' % (type(other), nanotime))


days = _converter.days
hours = _converter.hours
minutes = _converter.minutes
seconds = _converter.seconds
milliseconds = _converter.milliseconds
microseconds = _converter.microseconds
nanoseconds = _converter.nanoseconds
unixtime = _converter.unixtime
timestamp = _converter.timestamp
datetime = _converter.datetime
convert = _converter.convert
now = _converter.now


