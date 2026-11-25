import importlib.util
import json
from pathlib import Path
import sys

import pytest


def module():
    path=Path(__file__).parents[3]/"tools/tpu/instance.py"
    spec=importlib.util.spec_from_file_location("tpu_instance_tool",path)
    tool=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_instance_settings_are_scoped_to_host(tmp_path,monkeypatch):
    tool=module()
    config=tmp_path/"instance.json"
    config.write_text(json.dumps({"hostname":"host1","environment":{"TPU_HOST_BOUNDS":"1,1,1","TPU_LOG_DIR":"out/logs"}}))
    monkeypatch.setattr(tool.socket,"gethostname",lambda:"host1")
    monkeypatch.setattr(sys,"argv",["instance.py","--config",str(config),"--python","python-test","--","-m","amora","tpu","list"])
    calls=[]
    class Result:returncode=0
    def run(args,**kwargs):calls.append((args,kwargs));return Result()
    monkeypatch.setattr(tool.subprocess,"run",run)
    assert tool.main()==0
    assert calls[0][0]==["python-test","-m","amora","tpu","list"]
    assert calls[0][1]["env"]["TPU_HOST_BOUNDS"]=="1,1,1"
    monkeypatch.setattr(tool.socket,"gethostname",lambda:"host2")
    with pytest.raises(SystemExit):tool.main()
