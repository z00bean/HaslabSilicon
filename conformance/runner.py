"""Validate immutable corpus artifacts, then compare a simulator to stored answers."""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Zubin Bhuyan

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "conformance/corpus"
STATUS_KEYS = {"state", "last_accepted", "last_completed", "last_end", "reset_generation",
               "error_code", "error_seq", "error_space", "error_offset", "error_field"}
DIAGNOSTICS = {"error_space", "error_offset", "error_field"}
CAPACITY = {1:16384, 2:16384, 3:8192, 4:8192, 5:3072}


class CorpusError(ValueError):
    """Invalid schema, metadata, pin, artifact, or reference."""


class ConformanceError(AssertionError):
    """A valid fixture disagrees with an implementation."""


def require(condition, message):
    if not condition:
        raise CorpusError(message)


def fields(obj, required, optional=()):
    require(isinstance(obj, dict), "expected object")
    require(set(required) <= obj.keys() and obj.keys() <= set(required) | set(optional),
            "unknown or missing fields: " + str(set(obj) ^ set(required)))


def uint(value, maximum=0xFFFFFFFF):
    require(type(value) is int and 0 <= value <= maximum, "expected bounded unsigned integer")


def string(value):
    require(isinstance(value, str) and bool(value), "expected nonempty string")


def array(value):
    require(isinstance(value, list), "expected array")


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(CorpusError("nonfinite JSON")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorpusError(str(exc)) from exc


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_suite(root=DEFAULT_CORPUS):
    """Validate the entire suite before any simulator runs; return manifest/blobs."""
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    require(manifest_path.stat().st_size <= 4*1024*1024, "manifest too large")
    doc = load_json(manifest_path)
    fields(doc, {"schema_version", "profile", "abi", "contract_revision", "contract_sha256",
                 "abi_sha256", "byte_order", "provenance", "artifacts", "cases"})
    uint(doc["schema_version"])
    require(doc["schema_version"] == 1 and doc["profile"] == "haslab-v0-int8", "unsupported fixture profile/version")
    require(type(doc["abi"]) is list and len(doc["abi"]) == 2, "invalid ABI")
    for value in doc["abi"]: uint(value,255)
    require(doc["abi"] == [0,1] and doc["contract_revision"] == "0.2" and doc["byte_order"] == "little", "unsupported contract")
    for key, path in (("contract_sha256",ROOT/"docs/haslab-v0-contract.md"), ("abi_sha256",ROOT/"conformance/abi-v0.1.json")):
        require(doc[key] == sha(path.read_bytes()), "stale pin: " + key)
    string(doc["provenance"])
    require(isinstance(doc["artifacts"],dict) and bool(doc["artifacts"]), "missing artifacts")
    blobs = {}
    total = 0
    for name, entry in doc["artifacts"].items():
        require(re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+\.bin",name) is not None, "unsafe artifact path")
        fields(entry,{"bytes","sha256"}); uint(entry["bytes"],4*1024*1024)
        require(isinstance(entry["sha256"],str) and re.fullmatch(r"[0-9a-f]{64}",entry["sha256"]), "invalid digest")
        path = root / name
        require(path.resolve().is_relative_to(root), "artifact escapes corpus")
        require(path.stat().st_size == entry["bytes"], "artifact length mismatch: " + name)
        total += entry["bytes"]; require(total <= 64*1024*1024,"corpus too large")
        data = path.read_bytes()
        require(sha(data)==entry["sha256"], "artifact hash mismatch: " + name)
        blobs[name]=data
    array(doc["cases"]); require(bool(doc["cases"]), "empty corpus")
    ids=set(); used=set()

    def artifact(name, case_id):
        require(isinstance(name,str) and name in blobs and name.startswith(case_id+"/"), "missing or cross-case artifact")
        used.add(name)
        return blobs[name]

    for case in doc["cases"]:
        fields(case,{"id","derivation","tags","tensors","ext_bytes","initial","steps","commands"})
        string(case["id"]); require(re.fullmatch(r"[a-z0-9-]+",case["id"]) and case["id"] not in ids,"invalid/duplicate case ID")
        ids.add(case["id"]); string(case["derivation"])
        array(case["tags"]); require(bool(case["tags"]),"missing coverage tags")
        for tag in case["tags"]: string(tag)
        array(case["tensors"])
        for tensor in case["tensors"]:
            fields(tensor,{"layout","physical_shape","dtype"})
            require(tensor["layout"] in ("HWC8","KHWCI8") and tensor["dtype"] in ("I8","I32"),"unknown tensor layout/type")
            array(tensor["physical_shape"]); require(len(tensor["physical_shape"])==4,"expected rank four physical tensor")
            for dim in tensor["physical_shape"]: uint(dim,65535); require(dim>0,"zero dimension")
            require(tensor["physical_shape"][-1]==8,"expected eight physical lanes")
        uint(case["ext_bytes"],16*1024*1024); require(case["ext_bytes"]>0,"empty EXT")
        capacities={0:case["ext_bytes"], **CAPACITY}
        stream=artifact(case["commands"],case["id"])

        def spans(items, expected=False):
            array(items)
            for span in items:
                fields(span,{"space","offset","artifact"},{"mask"} if expected else ())
                uint(span["space"],5); uint(span["offset"])
                data=artifact(span["artifact"],case["id"])
                require(bool(data) and span["offset"]+len(data)<=capacities[span["space"]],"memory span outside region")
                if "mask" in span:
                    require(len(artifact(span["mask"],case["id"]))==len(data),"wrong memory mask length")
        spans(case["initial"])
        array(case["steps"]); require(bool(case["steps"]),"empty case")
        for step in case["steps"]:
            require(isinstance(step,dict),"step must be object")
            action=step.get("action")
            if action in ("submit","execute"):
                fields(step,{"action","offset","length","result"} if action=="submit" else {"action","offset","count","retired"})
                uint(step["offset"])
                if action=="submit":
                    uint(step["length"],129); uint(step["result"],3); size=step["length"]
                else:
                    uint(step["count"],65536); require(step["count"]>0,"empty execution")
                    uint(step["retired"],step["count"]); size=128*step["count"]
                require(step["offset"]+size<=len(stream),"command slice outside artifact")
            elif action=="run":
                fields(step,{"action","count","retired"}); uint(step["count"],65536)
                require(step["count"]>0,"empty run"); uint(step["retired"],step["count"])
            elif action=="reset": fields(step,{"action"})
            elif action=="check":
                fields(step,{"action","status","diagnostic_masks","memory"},{"allowed_error_codes"})
                fields(step["status"],STATUS_KEYS)
                for value in step["status"].values(): uint(value)
                require(step["status"]["state"]<=3 and step["status"]["error_code"]<=9,"unknown expected state/error")
                fields(step["diagnostic_masks"],DIAGNOSTICS)
                for value in step["diagnostic_masks"].values(): uint(value)
                if "allowed_error_codes" in step:
                    array(step["allowed_error_codes"]); require(bool(step["allowed_error_codes"]),"empty allowed errors")
                    for value in step["allowed_error_codes"]: uint(value,9)
                    require(step["status"]["error_code"] in step["allowed_error_codes"],"primary error missing from set")
                spans(step["memory"],True)
            else: raise CorpusError("unknown step action: "+str(action))
        require(case["steps"][-1]["action"]=="check","case must end in a check")
    require(used==set(blobs),"unreferenced artifacts")
    return doc,blobs


def snapshot(device):
    return dict(state=int(device.state), last_accepted=device.last_accepted,
                last_completed=device.last_completed, last_end=device.last_end,
                reset_generation=device.reset_generation, error_code=int(device.fault.code),
                error_seq=device.fault.sequence, error_space=device.fault.space,
                error_offset=device.fault.offset, error_field=device.fault.field)


def run_case(case, blobs, device_factory=None):
    # Imports here enforce an explicit boundary: loaders/recipes need no DUT.
    if device_factory is None:
        from haslab_sim import HaslabDevice
        device_factory=HaslabDevice
    device=device_factory(ext_bytes=case["ext_bytes"])
    for span in case["initial"]:
        device.memory.write(span["space"],span["offset"],blobs[span["artifact"]])
    stream=blobs[case["commands"]]
    for index,step in enumerate(case["steps"]):
        context="%s step %d"%(case["id"],index)

        def equal(actual,expected,label):
            if actual!=expected:
                raise ConformanceError("%s %s: expected %r, got %r"%(context,label,expected,actual))

        action=step["action"]
        if action=="submit":
            raw=stream[step["offset"]:step["offset"]+step["length"]]
            equal(int(device.submit(raw)),step["result"],"submit")
        elif action in ("run","execute"):
            if action=="execute":
                equal(device.queued_commands,0,"execute requires empty FIFO")
            retired=0
            for n in range(step["count"]):
                if action=="execute":
                    offset=step["offset"]+128*n
                    equal(int(device.submit(stream[offset:offset+128])),0,"execute submit")
                if not device.run_next(): break
                retired+=1
            equal(retired,step["retired"],"retired")
        elif action=="reset": device.reset()
        else:
            actual=snapshot(device)
            for key,value in step["status"].items():
                equal(type(actual[key]) is int and 0<=actual[key]<=0xFFFFFFFF,True,key+" fits u32")
                if key=="error_code" and "allowed_error_codes" in step:
                    equal(actual[key] in step["allowed_error_codes"],True,key)
                else:
                    if key in DIAGNOSTICS:
                        mask=step["diagnostic_masks"][key]
                        equal(actual[key]&mask,value&mask,key)
                    else:
                        equal(actual[key],value,key)
            for span in step["memory"]:
                expected=blobs[span["artifact"]]
                actual_bytes=device.memory.read(span["space"],span["offset"],len(expected))
                mask=blobs[span["mask"]] if "mask" in span else bytes([255])*len(expected)
                for n,(a,b,m) in enumerate(zip(actual_bytes,expected,mask)):
                    equal(a&m,b&m,"space %d byte %d"%(span["space"],span["offset"]+n))
    return device


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus",type=Path,default=DEFAULT_CORPUS)
    parser.add_argument("--case",help="run exactly one case ID")
    args=parser.parse_args()
    try:
        doc,blobs=load_suite(args.corpus)
        cases=[c for c in doc["cases"] if args.case is None or c["id"]==args.case]
        require(bool(cases),"unknown case")
        for case in cases:
            run_case(case,blobs)
            print("PASS "+case["id"])
        print("%d conformance cases passed"%len(cases))
    except (CorpusError,ConformanceError,OSError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__=="__main__": main()
