"""Exact cubin-byte verification for the disassembly text adapter."""
import struct
import pytest
from amora.backends.gcom_cuda.cubin_disassembly import legacy_text


def fixture():
    text=struct.pack('<QQQQ',0x123,0x456,0x789,0xabc)
    names=b'\x00.text.foo\x00.shstrtab\x00'
    table=64+len(text)+len(names)
    data=bytearray(table+3*64)
    data[:8]=b'\x7fELF\x02\x01\x01\x33'
    data[64:96]=text;data[96:96+len(names)]=names
    struct.pack_into('<Q',data,40,table)
    struct.pack_into('<HHH',data,58,64,3,2)
    struct.pack_into('<IIQQQQIIQQ',data,table+64,1,1,6,0,64,32,0,0,16,0)
    struct.pack_into('<IIQQQQIIQQ',data,table+128,11,3,0,0,96,len(names),0,0,1,0)
    disassembly=''' .section .text.foo,"ax",@progbits
/*0000*/ BRA `(.L_x_0) (*"BRANCH_TARGETS .L_x_0"*); /* 0x0000000000000123 */
/* 0x0000000000000456 */
.L_x_0:
/*0010*/ SHF.L.U32 R1, RZ, 0x1, RZ ; /* 0x0000000000000789 */
/* 0x0000000000000abc */
'''
    return bytes(data),disassembly


def test_legacy_rendering_preserves_encoding_and_resolves_only_branch_labels():
    data,sass=fixture();output,count=legacy_text(sass,data)
    assert count==2 and 'BRA 0x10' in output and 'SHF.L.U32' in output
    assert 'BRANCH_TARGETS' not in output and '`(' not in output
    assert 'Function : foo' in output and output.endswith('..........\n')


def test_tampered_instruction_or_missing_control_word_is_rejected():
    data,sass=fixture()
    with pytest.raises(ValueError,match='differs from original'):
        legacy_text(sass.replace('0000000000000123','0000000000000124'),data)
    with pytest.raises(ValueError,match='missing 128-bit'):
        legacy_text(sass.replace('/* 0x0000000000000456 */',''),data)


def test_incomplete_static_instruction_coverage_is_rejected():
    data,sass=fixture()
    with pytest.raises(ValueError,match='complete ELF text'):
        legacy_text(sass.split('.L_x_0:')[0].replace('.L_x_0','0x10'),data)
