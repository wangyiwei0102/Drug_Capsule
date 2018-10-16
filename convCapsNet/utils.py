import sys
import time
def print_time(start_time, f=None):
  """Take a start time, print elapsed duration, and return a new time."""
  s = "time %ds, %s." % ((time.time() - start_time), time.ctime())
  print(s)
  if f:
    f.write(s.encode("utf-8"))
    f.write(b"\n")
  sys.stdout.flush()
  return time.time()


def print_out(s, f=None, new_line=True):
  """Similar to print but with support to flush and output to a file."""
  if isinstance(s, bytes):
    s = s.decode("utf-8")

  if f:
    f.write(s.encode("utf-8"))
    if new_line:
      f.write(b"\n")

  # stdout
  print(s.encode("utf-8"), end='', file=sys.stdout)
  if new_line:
    sys.stdout.write("\n")
  sys.stdout.flush()



def print_hparams(hparams, skip_patterns=None, f=None):
  """Print hparams, can skip keys based on pattern."""
  values = hparams.values()
  for key in sorted(values.keys()):
    if not skip_patterns or all(
        [skip_pattern not in key for skip_pattern in skip_patterns]):
      print_out("  %s=%s" % (key, str(values[key])), f)