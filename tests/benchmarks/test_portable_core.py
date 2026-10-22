"""Packaged portable workloads; no hardware collection or performance claims."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

np = pytest.importorskip('numpy', reason='install the portable-core extra for workload tests')

from benchmark_generators import portable_core as package
from benchmark_generators.portable_core.base.core import digest


@pytest.mark.parametrize('tier,smoke,native', [(20,152,304),(24,168,336),(28,184,368)])
def test_case_fingerprints_preserve_upstream_contracts(tier,smoke,native):
    root=Path(package.__file__).parent
    fingerprints=json.loads((root/'SEMANTIC_FINGERPRINTS.json').read_text())['fingerprints']
    assert len(package.families(tier))==tier
    for profile,count in [('smoke',smoke),('native',native)]:
        cases=package.generate_cases(tier,profile)
        assert len(cases)==count
        assert digest(cases)==fingerprints[f'{tier}/{profile}']['case_payload_sha256']


def test_all_core28_numpy_oracles_and_tuple_outputs():
    backend=package.create_backend('numpy','cpu')
    for case in package.generate_cases(28):
        raw=package.make_inputs(case)
        actual=package.evaluate(case,backend,raw)
        expected=package.oracle(case,raw)
        assert isinstance(actual,tuple) and len(actual)==len(expected)
        matrix=any(n['op']=='matmul' for n in case['nodes'])
        for got,want in zip(actual,expected):
            np.testing.assert_allclose(backend.numpy(got),want,atol=5e-4 if matrix else 2e-5,
                                       rtol=5e-3 if matrix else 2e-4)


def test_torch_cpu_parity():
    pytest.importorskip('torch')
    backend=package.create_backend('torch','cpu')
    for case in package.generate_cases(28):
        raw=package.make_inputs(case)
        expected=package.oracle(case,raw)
        matrix=any(n['op']=='matmul' for n in case['nodes'])
        for got,want in zip(package.evaluate(case,backend,raw),expected):
            np.testing.assert_allclose(backend.numpy(got),want,atol=5e-4 if matrix else 2e-5,
                                       rtol=5e-3 if matrix else 2e-4)


@pytest.mark.parametrize('tier',[20,24,28])
def test_frozen_snapshot_and_overwrite_guard(tmp_path,tier):
    root=tmp_path/f'core{tier}'
    result=package.freeze(root,tier)
    assert result['status']=='FROZEN'
    package.verify_snapshot(root)
    with pytest.raises(RuntimeError): package.freeze(root,tier)


def test_import_and_discovery_do_not_require_array_libraries():
    script='''import builtins\noriginal=builtins.__import__\ndef blocked(name,*args,**kwargs):\n    if name.split('.')[0] in ('numpy','torch','jax'): raise ImportError(name)\n    return original(name,*args,**kwargs)\nbuiltins.__import__=blocked\nfrom benchmark_generators.portable_core import describe\nassert describe(28)['tier']==28\n'''
    subprocess.run([sys.executable,'-c',script],check=True)


def test_license_and_no_accorde_imports():
    root=Path(package.__file__).parent
    assert 'Apache License' in (root/'LICENSE.Apache-2.0').read_text()
    assert 'namespace relocation' in (root/'NOTICE').read_text()
    assert not any(k=='cross_isa_campaign' or k.startswith('cross_isa_campaign.') or
                   k=='cross_isa_expansion' or k.startswith('cross_isa_expansion.') for k in sys.modules)
    assert package.describe()['native_performance_qualified'] is False
