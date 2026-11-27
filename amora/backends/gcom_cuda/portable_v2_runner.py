"""Run one replay child and retain its exact wait4 resource usage."""
import json
import os
from pathlib import Path
import sys


def main():
    destination, command = sys.argv[1], sys.argv[2:]
    pid = os.fork()
    if pid == 0:
        try: os.execvpe(command[0], command, os.environ)
        except OSError as exc:
            print(str(exc), file=sys.stderr, flush=True)
            os._exit(127)
    _, status, usage = os.wait4(pid, 0)
    code = os.waitstatus_to_exitcode(status)
    Path(destination).write_text(json.dumps(dict(exit_code=code, max_rss_kib=usage.ru_maxrss,
        user_seconds=usage.ru_utime, system_seconds=usage.ru_stime,
        major_page_faults=usage.ru_majflt, minor_page_faults=usage.ru_minflt)) + "\n")
    return code if code >= 0 else 128-code


if __name__ == "__main__": sys.exit(main())
