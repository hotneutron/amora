"""Build a separately identified 64-program batch driver without changing math."""
from pathlib import Path
import subprocess

from amora.backends.gcom_cuda.portable_v2 import sha,write,utc


def batch_source(original):
    marker="int main(int argc,char **argv)"
    if marker not in original:raise ValueError("unknown portable driver main boundary")
    prefix,main=original.split(marker,1)
    replacements={
        "samples=0,warmup=3;":"samples=0,warmup=3,batch=1;",
        'else if(name=="--samples")samples=std::stoi(value);':
            'else if(name=="--batch")batch=std::stoi(value);else if(name=="--samples")samples=std::stoi(value);',
        'if(chosen.empty() || samples<0 || warmup<0)':
            'if(chosen.empty() || samples<0 || warmup<0 || (batch!=1 && batch!=64))',
        "auto &c=cases[index];":"auto &c=cases[index];\n      auto execute_batch=[&](){for(int repetition=0;repetition<batch;repetition++)execute(c);};",
        "for(int w=0;w<warmup;w++)execute(c);":"for(int w=0;w<warmup;w++)execute_batch();",
        'float ms;check_cuda(cudaEventRecord(begin),"event begin");execute(c);':
            'float ms;check_cuda(cudaEventRecord(begin),"event begin");execute_batch();',
    }
    for before,after in replacements.items():
        if main.count(before)!=1:raise ValueError("driver batch adapter boundary changed: "+before)
        main=main.replace(before,after)
    return prefix+marker+main


def build_batch(root,original,contract):
    import hashlib,json
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    original=Path(original)
    source=batch_source(original.read_text())
    digest=hashlib.sha256(source.encode()).hexdigest()
    folder=root/digest;folder.mkdir(exist_ok=True)
    receipt=folder/"build.json"
    if receipt.exists():
        value=json.loads(receipt.read_text())
        if sha(value["binary"])!=value["binary_sha256"]:raise ValueError("batch driver changed")
        return value
    path=folder/"portable-v2-batch.cu";binary=folder/"portable-v2-batch"
    if (folder/"started.json").exists():raise RuntimeError("batch build previously attempted; inspect its receipt")
    path.write_text(source)
    argv=["/usr/local/cuda-13.3/bin/nvcc","-ccbin","/usr/bin/g++-11",*contract["flags"],
          "-DAMORA_OP_MASK=2147483647ULL",str(path),"-o",str(binary)]
    write(folder/"started.json",dict(timestamp=utc(),argv=argv,user_prompt="execute the plan"))
    with (folder/"build.log").open("x") as output:
        result=subprocess.run(argv,stdout=output,stderr=subprocess.STDOUT)
    if result.returncode!=0:raise RuntimeError("batch driver build failed; see "+str(folder/"build.log"))
    value=dict(binary=str(binary),binary_sha256=sha(binary),source=str(path),source_sha256=sha(path),
        original_source_sha256=sha(original),timestamp=utc(),argv=argv,
        realization="cuda-explicit-node-boundaries-batch64-v1",host_compiler="GCC 11",cuda="13.3",
        adapter="one validation graph; 3 batches of 64 warmups; each sample executes 64 independent prepared graphs",
        kernel_math_prefix_sha256=hashlib.sha256(original.read_text().split("int main(",1)[0].encode()).hexdigest(),
        native_qualification="pending; no single-program timing transferred")
    write(receipt,value)
    return value
