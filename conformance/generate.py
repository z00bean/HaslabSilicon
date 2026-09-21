"""Deterministic corpus recipes. No simulator, NumPy, or golden-model imports.

Expected bytes and control states are independently authored from the contract.
This script serializes them; it never asks an implementation for expected output.
"""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import argparse
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def command(opcode, sequence, payload=(), flags=0, version=0x00010000, reserved=0):
    # Independent encoder; deliberately does not consume executable ABI constants.
    return struct.pack("<32I", version | opcode, flags, sequence, reserved,
                       *payload, *([0] * (28 - len(payload))))


def i8(values):
    return struct.pack("<" + "b" * len(values), *values)


def i32(values):
    return struct.pack("<" + "i" * len(values), *values)


def params(bias=None, multiplier=None, shift=None):
    arrays = [list(v or []) + [0] * (8 - len(v or [])) for v in (bias, multiplier, shift)]
    return b"".join(struct.pack("<iIII", arrays[0][i], arrays[1][i], arrays[2][i], 0) for i in range(8))


def status(accepted, completed, end=0, state=1, code=0, seq=0, generation=0):
    return dict(state=state, last_accepted=accepted, last_completed=completed,
                last_end=end, reset_generation=generation, error_code=code, error_seq=seq,
                error_space=0xFFFFFFFF, error_offset=0, error_field=0xFFFFFFFF)


class Case:
    def __init__(self, name, derivation, tags, tensors=None, ext_bytes=256):
        self.name = name
        self.files = {}
        self.stream = bytearray()
        self.doc = dict(id=name, derivation=derivation, tags=tags,
                        tensors=tensors or [], ext_bytes=ext_bytes, initial=[], steps=[])

    def blob(self, name, data):
        path = self.name + "/" + name
        self.files[path] = bytes(data)
        return path

    def initial(self, space, offset, data):
        path = self.blob("initial-%d.bin" % len(self.doc["initial"]), data)
        self.doc["initial"].append(dict(space=space, offset=offset, artifact=path))

    def submit(self, raw, result=0):
        offset = len(self.stream)
        self.stream.extend(raw)
        self.doc["steps"].append(dict(action="submit", offset=offset, length=len(raw), result=result))

    def execute(self, records, retired=None):
        offset = len(self.stream)
        self.stream.extend(b"".join(records))
        self.doc["steps"].append(dict(action="execute", offset=offset, count=len(records),
                                      retired=len(records) if retired is None else retired))

    def check(self, expected, memory=()):
        spans = []
        for space, offset, data in memory:
            path = self.blob("expected-%d.bin" % len(self.files), data)
            spans.append(dict(space=space, offset=offset, artifact=path))
        self.doc["steps"].append(dict(action="check", status=expected,
                                      diagnostic_masks=dict(error_space=0, error_offset=0, error_field=0),
                                      memory=spans))

    def finish(self):
        self.doc["commands"] = self.blob("commands.bin", self.stream)
        return self.doc, self.files


def tensor(layout, shape):
    return dict(layout=layout, physical_shape=shape, dtype="I8")


def recipes():
    cases = []

    c = Case("dma-fill-copy", "FILL writes 0x55. DMA gathers two 8-byte EXT rows at 0 and 16. COPY moves the 16 gathered bytes to OUTPUT; DMA scatters them at EXT 64 and 80. Guard bytes must survive.", ["DMA_COPY2D", "FILL8", "COPY2D", "FENCE", "END", "guards"])
    c.initial(0, 0, bytes(range(24)))
    c.initial(0, 64, b"!" * 24)
    c.execute([command(2, 1, [1, 0, 24, 85]), command(1, 2, [0, 0, 1, 0, 8, 2, 16, 8, 0]),
               command(36, 3, [1, 0, 4, 0, 16, 1, 16, 16, 0]), command(1, 4, [4, 0, 0, 64, 8, 2, 8, 16, 0]),
               command(48, 5), command(49, 6)])
    c.check(status(6, 6, 6), [(1, 0, bytes(range(8)) + bytes(range(16, 24)) + b"U" * 8),
                              (4, 0, bytes(range(8)) + bytes(range(16, 24))),
                              (0, 64, bytes(range(8)) + b"!" * 8 + bytes(range(16, 24)))])
    cases.append(c)

    c = Case("conv-raw", "Lane 0: 1*2+2*3+3*4+5=25; lane 1: 1+2+3-1=5. Six padding lanes remain zero.", ["CONV_I8", "EPILOGUE", "padding", "bias"], [tensor("HWC8", [1, 1, 1, 8]), tensor("KHWCI8", [1, 1, 8, 8])])
    c.initial(1, 0, i8([1, 2, 3, 0, 0, 0, 0, 0]))
    c.initial(2, 0, i8([2, 1] + [0]*6 + [3, 1] + [0]*6 + [4, 1] + [0]*46))
    c.initial(5, 0, params(bias=[5, -1]))
    c.execute([command(16, 1, [0, 0, 0, 1, 1, 2, 3, 0, 3, 1, 1], 3)])
    c.check(status(1, 1, state=2), [(3, 0, i32([20, 6] + [0]*6))])
    c.execute([command(17, 2, [0, 0, 0, 0, 0]), command(49, 3)])
    c.check(status(3, 3, 3), [(4, 0, i32([25, 5] + [0]*6))])
    cases.append(c)

    c = Case("conv-chunks", "First eight channels contribute 8. Final two channels contribute 2*4+3*5=23. ACC is 31 before bias; no intermediate requantization.", ["CONV_I8", "EPILOGUE", "chunks", "FENCE"], [tensor("HWC8", [1, 1, 1, 8])])
    c.initial(1, 0, bytes([1]*8 + [2, 3] + [0]*6))
    c.initial(2, 0, bytes(([1]+[0]*7)*8 + [4]+[0]*7 + [5]+[0]*55))
    c.initial(5, 0, params())
    c.execute([command(16, 1, [0, 0, 0, 1, 1, 1, 8, 0, 10, 1, 1], 1), command(48, 2)])
    c.check(status(2, 2, state=2), [(3, 0, i32([8]+[0]*7))])
    c.execute([command(16, 3, [8, 64, 0, 1, 1, 1, 2, 8, 10, 1, 1], 2), command(17, 4, [0, 0, 0, 0, 0]), command(49, 5)])
    c.check(status(5, 5, 5), [(4, 0, i32([31]+[0]*7))])
    cases.append(c)

    for name, source, out_hw, stride, expected in (
        ("conv3-stride2", list(range(1, 26)), 2, 2, [63, 81, 153, 171]),
        ("conv3-zero-halo", [0, 0, 0, 0, 7, 0, 0, 0, 0], 1, 1, [7]),
    ):
        c = Case(name, "All nine valid weights are 1. Sum each explicit 3x3 window; stride2 window sums are 63,81,153,171. The zero-halo case sums to 7.", ["CONV_I8", "EPILOGUE", "halo", "stride"], [tensor("HWC8", [5 if stride == 2 else 3, 5 if stride == 2 else 3, 1, 8])])
        c.initial(1, 0, i8([v for x in source for v in [x]+[0]*7]))
        c.initial(2, 0, bytes(([1]+[0]*63)*9))
        c.initial(5, 0, params())
        c.execute([command(16, 1, [0, 0, 0, out_hw, out_hw, 1, 1, 0, 1, 3, stride], 3), command(17, 2, [0, 0, 0, 0, 0]), command(49, 3)])
        c.check(status(3, 3, 3), [(4, 0, i32([v for x in expected for v in [x]+[0]*7]))])
        cases.append(c)

    for op in ("map", "linear-epilogue"):
        c = Case(op+"-ties", "RNE(5/2,7/2,-5/2,-7/2)=[2,4,-2,-4]; 127*2 and -128*2 saturate; M=0 zeroes the final lanes.", ["MAP_I8" if op == "map" else "EPILOGUE", "ties", "saturation", "zero-multiplier"], [tensor("HWC8", [1, 1, 1, 8])])
        values = [5, 7, -5, -7, 127, -128, 1, -1]
        c.initial(5, 0, params(bias=values if op != "map" else None, multiplier=[1,1,1,1,2,2,0,0], shift=[1,1,1,1,0,0,0,0]))
        if op == "map":
            c.initial(1, 0, i8(values))
            records = [command(32, 1, [0, 0, 1, 1, 8, 0])]
        else:
            c.initial(1, 0, bytes(8)); c.initial(2, 0, bytes(64))
            records = [command(16, 1, [0,0,0,1,1,8,1,0,1,1,1], 3), command(17, 2, [0,0,0,1,0])]
        records.append(command(49, len(records)+1)); c.execute(records)
        c.check(status(len(records), len(records), len(records)), [(4,0,i8([2,4,-2,-4,127,-128,0,0]))])
        cases.append(c)

    c = Case("add-single-round", "Sum both integers before dividing by 2. Numerators [2,3,-2,-3,254,-256,7,-7] round/saturate to [1,2,-1,-2,127,-128,4,-4].", ["ADD_I8", "ties", "saturation"], [tensor("HWC8", [1,1,1,8])])
    c.initial(1,0,i8([1,3,-1,-3,127,-128,5,-5,1,0,-1,0,127,-128,2,-2]))
    c.initial(5,0,params(multiplier=[1]*8,shift=[1]*8))
    c.execute([command(33,1,[0,8,0,1,1,8,0,0]),command(49,2)])
    c.check(status(2,2,2),[(4,0,i8([1,2,-1,-2,127,-128,4,-4]))]); cases.append(c)

    c = Case("silu-lookup-boundaries", "Synthetic LUT tests addressing, not SiLU accuracy. Bias/2 indices clamp at -512 and 511; ties select -2,-4,2,4,0,0. Explicit table entries return [-128,127,-2,-4,2,4,0,0].", ["EPILOGUE", "lut", "ties", "clamp"], [tensor("HWC8", [1,1,1,8])])
    c.initial(1,0,bytes(8)); c.initial(2,0,bytes(64))
    c.initial(5,0,params(bias=[-2147483648,2147483647,-5,-7,5,7,0,1],multiplier=[1]*8,shift=[1]*8))
    lut = bytearray(1024)
    for index,value in {0:128,1023:127,510:254,508:252,514:2,516:4}.items(): lut[index]=value
    c.initial(5,128,lut)
    c.execute([command(16,1,[0,0,0,1,1,8,1,0,1,1,1],3),command(17,2,[0,0,0,2,128]),command(49,3)])
    c.check(status(3,3,3),[(4,0,i8([-128,127,-2,-4,2,4,0,0]))]); cases.append(c)

    c = Case("pool-negative-halo", "The 5x5 patch is -128 except centre=-7; maximum=-7. Invalid output lanes are zero, not -128.", ["MAXPOOL5_I8", "halo", "padding"], [tensor("HWC8",[5,5,1,8])])
    source=bytearray([128]*200); source[12*8]=249
    c.initial(1,0,source); c.execute([command(34,1,[0,0,1,1,1]),command(49,2)])
    c.check(status(2,2,2),[(4,0,i8([-7]+[0]*7))]); cases.append(c)

    c = Case("upsample-padding", "Two input pixels -3 and 7 replicate to two rows [-3,-3,7,7]. Nonzero input padding is masked.", ["UPSAMPLE2_I8","padding"], [tensor("HWC8",[1,2,1,8])])
    c.initial(1,0,i8([-3]+[99]*7+[7]+[99]*7))
    c.execute([command(35,1,[0,0,1,2,1]),command(49,2)])
    c.check(status(2,2,2),[(4,0,i8([v for x in [-3,-3,7,7]*2 for v in [x]+[0]*7]))]); cases.append(c)

    c = Case("copy-interleaved", "Source rows [0,8) and [16,24) do not intersect destination rows [8,16) and [24,32).", ["COPY2D","alias"])
    c.initial(1,0,b"abcdefgh........ijklmnop........")
    c.execute([command(36,1,[1,0,1,8,8,2,16,16,0]),command(49,2)])
    c.check(status(2,2,2),[(1,0,b"abcdefghabcdefghijklmnopijklmnop")]); cases.append(c)

    c = Case("fifo-retry", "Eight FENCE records fill the FIFO; malformed input returns BUSY first. After one retirement it returns INVALID. Sequence 9 can then be retried unchanged.", ["fifo","framing","FENCE","END"])
    for seq in range(1,9): c.submit(command(48,seq))
    c.submit(b"bad",1); c.doc["steps"].append(dict(action="run",count=1,retired=1))
    c.submit(b"bad",3); c.submit(command(49,9))
    c.doc["steps"].append(dict(action="run",count=8,retired=8))
    c.check(status(9,9,9)); cases.append(c)

    c = Case("reset-drops-queue", "Reset discards a queued fill, clears counters, increments generation, preserves untouched EXT. Local reset bytes are unspecified and not compared; a new fill initializes them before observation.", ["reset","END","FILL8"])
    c.initial(0,0,b"persist!"); c.submit(command(2,1,[1,0,8,99]))
    c.doc["steps"].append(dict(action="reset")); c.check(status(0,0,generation=1),[(0,0,b"persist!")])
    c.execute([command(2,1,[1,0,8,7]),command(49,2)])
    c.check(status(2,2,2,generation=1),[(1,0,b"\x07"*8)]); cases.append(c)

    c = Case("framing-rejections", "Malformed length, wrong ABI, sequence zero/gap, reserved header, and illegal flags reject without consuming sequence 1 or changing execution fault state.", ["framing","version","sequence","flags"])
    for raw in (bytes(127),command(48,1,version=0x00020000),command(48,0),command(48,2),command(48,1,reserved=1),command(48,1,flags=1)):
        c.submit(raw,3); c.check(status(0,0))
    c.execute([command(49,1)]); c.check(status(1,1,1)); cases.append(c)

    c = Case("sticky-unknown-opcode", "Unknown opcode is accepted, then faults at seq2. Seq3 is queued but must not overwrite the INPUT sentinel. Subsequent submission is BLOCKED until reset.", ["BAD_OPCODE","sticky-fault"])
    c.initial(1,0,b"sentinel")
    for raw in (command(48,1),command(65535,2),command(2,3,[1,0,8,99])): c.submit(raw)
    c.doc["steps"].append(dict(action="run",count=3,retired=1)); c.submit(command(49,4),2)
    c.check(status(3,1,state=3,code=1,seq=2),[(1,0,b"sentinel")]); cases.append(c)

    invalid = [
        ("dma-alignment",command(1,1,[0,1,1,0,8,1,8,8,0]),3),
        ("dma-bounds",command(1,1,[0,256,1,0,8,1,8,8,0]),4),
        ("dma-wide-bounds",command(1,1,[0,0xFFFFFFF8,1,0,8,2,8,8,0]),4),
        ("dma-space",command(1,1,[1,0,4,0,8,1,8,8,0]),5),
        ("dma-single-row-stride",command(1,1,[0,0,1,0,8,1,16,8,0]),2),
        ("dma-zero-size",command(1,1,[0,0,1,0,0,1,0,0,0]),2),
        ("copy-cross-row-overlap",command(36,1,[1,0,1,16,8,2,16,16,0]),2),
        ("epilogue-without-context",command(17,1,[0,0,0,0,0]),6),
        ("conv-missing-first",command(16,1,[0,0,0,1,1,1,8,8,16,1,1],2),6),
        ("conv-bounds",command(16,1,[16384,0,0,1,1,1,1,0,1,1,1],3),4),
    ]
    for name,raw,code in invalid:
        c=Case(name,"One isolated invalid operand/context must produce fault code %d before changing INPUT sentinel; no command retires."%code,["negative",name])
        c.initial(1,0,b"sentinel"); c.execute([raw],retired=0)
        c.check(status(1,0,state=3,code=code,seq=1),[(1,0,b"sentinel")]); cases.append(c)

    payloads={1:[0,0,1,0,8,1,8,8,0],2:[1,0,8,0],16:[0,0,0,1,1,1,1,0,1,1,1],
              17:[0,0,0,0,0],32:[0,0,1,1,1,0],33:[0,0,0,1,1,1,0,0],
              34:[0,0,1,1,1],35:[0,0,1,1,1],36:[1,0,4,0,8,1,8,8,0],48:[],49:[]}
    for opcode,payload in payloads.items():
        c=Case("reserved-payload-%02x"%opcode,"P27 is reserved for every v0 opcode. Other operands and context are valid; nonzero P27 alone causes BAD_FIELD.",["reserved","negative"])
        c.initial(0,0,bytes(8)); c.initial(1,0,bytes(200)); c.initial(2,0,bytes(64)); c.initial(5,0,params()); c.initial(4,0,b"sentinel")
        seq=1
        if opcode==17:
            c.execute([command(16,1,[0,0,0,1,1,1,1,0,1,1,1],3)]); seq=2
        c.execute([command(opcode,seq,payload+[0]*(27-len(payload))+[1],3 if opcode==16 else 0)],retired=0)
        c.check(status(seq,seq-1,state=3,code=2,seq=seq),[(4,0,b"sentinel")])
        cases.append(c)

    for name,tail in (("end-open-context",command(49,2)),("utility-open-context",command(35,2,[0,0,1,1,1])),("conv-wrong-continuation",command(16,2,[0,0,0,1,1,1,8,0,16,1,1],0))):
        c=Case(name,"A first chunk opens ACC with 8 of 16 channels consumed; the following command violates context ownership or continuation.",["context","negative"])
        c.initial(1,0,bytes(8)); c.initial(2,0,bytes(64))
        c.execute([command(16,1,[0,0,0,1,1,1,8,0,16,1,1],1),tail],retired=1)
        c.check(status(2,1,state=3,code=6,seq=2),[(3,0,bytes(32))]); cases.append(c)

    c=Case("add-shift-mismatch","ADD lane 0 has shifts 0 and 1, which must fault before writing OUTPUT.",["parameters","negative"])
    c.initial(1,0,bytes(16)); c.initial(5,0,params(multiplier=[1])+params(multiplier=[1],shift=[1])); c.initial(4,0,b"sentinel")
    c.execute([command(33,1,[0,8,0,1,1,1,0,128])],retired=0)
    c.check(status(1,0,state=3,code=2,seq=1),[(4,0,b"sentinel")]); cases.append(c)

    c=Case("map-padding-record","MAP has only one valid lane, so lane 1 multiplier=1 is an invalid padding record.",["parameters","padding","negative"])
    c.initial(1,0,bytes(8)); c.initial(5,0,params(multiplier=[1,1])); c.initial(4,0,b"sentinel")
    c.execute([command(32,1,[0,0,1,1,1,0])],retired=0)
    c.check(status(1,0,state=3,code=2,seq=1),[(4,0,b"sentinel")]); cases.append(c)

    for name,mode,lut_offset,output_offset,code in (("bias-overflow",0,0,0,7),("lut-overlap",2,0,0,2),("epilogue-output-preflight",0,0,8192,4)):
        c=Case(name,"CONV produces 1; adding INT32_MAX would overflow. Structural LUT/output errors must win before arithmetic. No memory is asserted after a numeric fault.",["EPILOGUE","overflow" if code==7 else "preflight","negative"])
        c.initial(1,0,b"\x01"+bytes(7)); c.initial(2,0,b"\x01"+bytes(63)); c.initial(5,0,params(bias=[2147483647])); c.initial(4,0,b"sentinel")
        c.execute([command(16,1,[0,0,0,1,1,1,1,0,1,1,1],3),command(17,2,[0,output_offset,0,mode,lut_offset])],retired=1)
        c.check(status(2,1,state=3,code=code,seq=2),[] if code==7 else [(4,0,b"sentinel")]); cases.append(c)

    c=Case("accumulator-overflow","Each -128*-128 term is 16384; 3*3*32=288 terms per chunk. 455 chunks give 2146959360. Chunk 456 crosses INT32_MAX at term 131072 overall and must not retire. Failed ACC/output bytes are unspecified.",["CONV_I8","overflow","chunks","negative"])
    c.initial(1,0,bytes([128])*288)
    c.initial(2,0,bytes([v for _ in range(288) for v in [128]+[0]*7]))
    records=[command(16,n+1,[0,0,0,1,1,1,32,n*32,14600,3,1],1 if n==0 else 0) for n in range(456)]
    c.execute(records,retired=455); c.check(status(456,455,state=3,code=7,seq=456)); cases.append(c)
    return cases


def render():
    files = {}
    docs = []
    for case in recipes():
        doc, blobs = case.finish()
        if doc["id"] in {d["id"] for d in docs}: raise ValueError("duplicate case")
        docs.append(doc); files.update(blobs)
    manifest = dict(schema_version=1, profile="haslab-v0-int8", abi=[0,1], contract_revision="0.2",
                    contract_sha256=digest((ROOT/"docs/haslab-v0-contract.md").read_bytes()),
                    abi_sha256=digest((ROOT/"conformance/abi-v0.1.json").read_bytes()),
                    byte_order="little", provenance="Independent manual contract derivations; serialized by conformance/generate.py; no implementation imports; not external review.",
                    artifacts={name:dict(bytes=len(data),sha256=digest(data)) for name,data in sorted(files.items())}, cases=docs)
    files["manifest.json"]=(json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode()
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true",help="explicitly regenerate checked-in assets")
    args=parser.parse_args()
    root=ROOT/"conformance/corpus"
    files=render()
    if args.write:
        for name,data in files.items():
            path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
    else:
        actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        if actual != set(files): raise SystemExit("corpus file list differs; review before --write")
        for name,data in files.items():
            if (root/name).read_bytes()!=data: raise SystemExit("corpus differs: "+name)
    print("%d independently authored cases; corpus %s"%(len(recipes()),"written" if args.write else "reproduces byte-for-byte"))


if __name__=="__main__": main()
