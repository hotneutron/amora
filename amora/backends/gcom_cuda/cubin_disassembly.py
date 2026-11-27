"""Render CUDA Std ELF cubins for GCoM's legacy cuobjdump text reader.

CUDA 13.3 cuobjdump rejects some driver-loaded Std ELF cubins which its
nvdisasm accepts. This adapter changes text framing only. Every exported
128-bit instruction is verified against the original ELF section bytes.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys


def elf_text_sections(data):
    if data[:6] != b'\x7fELF\x02\x01':raise ValueError('not ELF64 little endian')
    offset=struct.unpack_from('<Q',data,40)[0]
    size,count,strings=struct.unpack_from('<HHH',data,58)
    if size<64 or offset+size*count>len(data):raise ValueError('invalid ELF section table')
    headers=[struct.unpack_from('<IIQQQQIIQQ',data,offset+i*size) for i in range(count)]
    sh=headers[strings];names=data[sh[4]:sh[4]+sh[5]]
    sections={}
    for h in headers:
        name=names[h[0]:].split(b'\x00',1)[0].decode()
        if name.startswith('.text.'):
            payload=data[h[4]:h[4]+h[5]]
            if len(payload)!=h[5]:raise ValueError('truncated ELF text section')
            sections[name[6:]]=payload
    if not sections:raise ValueError('ELF has no CUDA text functions')
    return sections


def legacy_text(disassembly, data):
    sections=elf_text_sections(data)
    functions={};current=None;labels=[]
    lines=disassembly.splitlines()
    for index,line in enumerate(lines):
        section=re.search(r'\.section\s+\.text\.([^,\s]+)',line)
        if section:
            current=section[1];functions.setdefault(current,dict(instructions=[],labels={}))
            labels=[]
        if current is None:continue
        label=re.fullmatch(r'\s*(\.L[^\s:]+):\s*',line)
        if label:labels.append(label[1])
        instruction=re.match(r'\s*/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*/\*\s*(0x[0-9a-fA-F]{16})\s*\*/\s*$',line)
        if not instruction:continue
        pc=int(instruction[1],16)
        high=re.search(r'/\*\s*(0x[0-9a-fA-F]{16})\s*\*/',lines[index+1])
        if not high:raise ValueError('missing 128-bit control word')
        encoded=struct.pack('<QQ',int(instruction[3],16),int(high[1],16))
        if sections[current][pc:pc+16]!=encoded:
            raise ValueError('disassembly encoding differs from original cubin')
        for label in labels:functions[current]['labels'][label]=pc
        labels=[]
        functions[current]['instructions'].append((pc,instruction[2],instruction[3],high[1]))
    output=['code for sm_90']
    count=0
    for name,function in functions.items():
        instructions=function['instructions']
        if not instructions:continue
        if [pc for pc,*_ in instructions] != list(range(0,len(sections[name]),16)):
            raise ValueError('disassembly does not cover the complete ELF text section')
        output.append('Function : '+name)
        output.append('.headerflags @"EF_CUDA_SM90"')
        for pc,text,low,high in instructions:
            text=re.sub(r'\(\*".*?"\*\)', '', text)
            def replace_label(match):
                label=match[0]
                if label not in function['labels']:raise ValueError('unresolved branch label '+label)
                return hex(function['labels'][label])
            text=re.sub(r'\.L_[\w.$]+',replace_label,text)
            text=re.sub(r'`\((0x[0-9a-fA-F]+)\)',r'\1',text)
            output.extend([f'/*{pc:04x}*/ {text} /*{low}*/',f'/*{high}*/'])
            count+=1
        output.append('..........')
    if not count:raise ValueError('no verified instructions')
    return '\n'.join(output)+'\n',count


def main():
    args=sys.argv[1:]
    real=Path(os.environ.get('AMORA_REAL_CUOBJDUMP','/usr/local/cuda-13.3/bin/cuobjdump'))
    original=subprocess.run([str(real),*args],capture_output=True)
    if len(args)!=2 or args[0]!='-sass' or b'Function :' in original.stdout:
        sys.stdout.buffer.write(original.stdout);sys.stderr.buffer.write(original.stderr)
        return original.returncode
    cubin=Path(args[1]);data=cubin.read_bytes()
    if data[:4]!=b'\x7fELF' or data[7]!=0x33:
        sys.stdout.buffer.write(original.stdout);sys.stderr.buffer.write(original.stderr)
        return original.returncode
    nvdisasm=Path(os.environ.get('AMORA_REAL_NVDISASM','/usr/local/cuda-13.3/bin/nvdisasm'))
    result=subprocess.run([str(nvdisasm),'-c','-hex',str(cubin)],capture_output=True,text=True,check=True)
    text,count=legacy_text(result.stdout,data)
    receipt=dict(binary=str(cubin),binary_sha256=hashlib.sha256(data).hexdigest(),
        original_cuobjdump_exit_code=original.returncode,verified_instructions=count,
        rendered_sha256=hashlib.sha256(text.encode()).hexdigest(),
        nvdisasm_sha256=hashlib.sha256(nvdisasm.read_bytes()).hexdigest(),
        scope='legacy text framing and resolved branch labels; all instruction encodings verified against original ELF bytes',
        timing_model_changed=False)
    destination=os.environ.get('AMORA_DISASSEMBLY_RECEIPTS')
    if destination:
        folder=Path(destination);folder.mkdir(parents=True,exist_ok=True)
        (folder/(cubin.name+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    sys.stderr.write('AMORA verified Std ELF disassembly adapter: '+str(count)+' instructions\n')
    sys.stdout.write(text)
    return 0


if __name__=='__main__':sys.exit(main())
