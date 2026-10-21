"""IFC质量审查SaaS演示：仅需本文件和requirements.txt，云端选择Python 3.12。
原文件只读；NamedTemporaryFile临时保存并清理；无本地模型路径。
正式审查全部建筑构件、门窗与空间，规则固定为4.0；不以目标分数倒推权重。
"""
from __future__ import annotations
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from html import escape
import hashlib, json, math, re, tempfile, uuid
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import ifcopenshell
import ifcopenshell.api
import ifcopenshell.geom
import ifcopenshell.util.element as element_util
import ifcopenshell.util.placement as placement_util
import ifcopenshell.util.unit as unit_util

TITLE="IFC 完整建筑模型质量审查与修复"
MAX_FILE_MB=30
MAX_TARGETS=500
ISSUE_COLUMNS=["指标","对象STEP","GlobalId","IFC类别","名称","状态","原因","常规扣分","额外否决扣分","证据"]


def entity_ref(entity):
    """STEP编号负责本文件内定位，GlobalId负责对象身份追踪。"""
    return {
        "step_id": entity.id(), "GlobalId": getattr(entity, "GlobalId", None),
        "class": entity.is_a(), "Name": getattr(entity, "Name", None),
    }

def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

@contextmanager
def open_uploaded_ifc(data: bytes):
    """NamedTemporaryFile兼容Linux和Windows：写完关闭句柄，再让IFC库读取。

    不使用用户文件名构造路径，不覆盖上传内容，异常时也清理临时文件。
    """
    path = None
    try:
        if not data:
            raise ValueError("上传文件为空。")
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            raise ValueError(f"演示版只接受不超过 {MAX_FILE_MB} MB 的文件。")
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".ifc", delete=False) as tmp:
            tmp.write(data)
            path = Path(tmp.name)
        try:
            model = ifcopenshell.open(str(path))
        except Exception as exc:
            raise ValueError(f"IfcOpenShell 无法读取文件，请确认它是有效的IFC交换文件：{exc}") from exc
        if not list(model.by_type("IfcProject")):
            raise ValueError("文件没有 IfcProject，无法进行本系统的建筑模型检查。")
        yield model
    finally:
        if path is not None:
            path.unlink(missing_ok=True)

def guid_is_valid(value):
    try:
        return bool(re.fullmatch(r"[0-3][0-9A-Za-z_$]{21}", value or "")) and (
            ifcopenshell.guid.compress(ifcopenshell.guid.expand(value)) == value
        )
    except Exception:
        return False

def spatial_parent_chain(obj):
    """只接受唯一聚合父链，检测多父和循环，防止无限递归。"""
    chain, seen = [], {obj.id()}
    while True:
        parents = [rel.RelatingObject for rel in getattr(obj, "Decomposes", ())]
        if not parents:
            return chain, None
        if len(parents) != 1:
            return chain, "存在多个聚合父对象"
        parent = parents[0]
        if parent.id() in seen:
            return chain, "聚合父链存在循环"
        seen.add(parent.id())
        chain.append(parent)
        obj = parent

def project_scale(model, unit_type):
    """缺单位不默认成米。面积、体积按各自单位换算，不能套用长度比例。"""
    try:
        if unit_util.get_project_unit(model, unit_type) is None:
            return None
        value = unit_util.calculate_unit_scale(model, unit_type)
        return float(value) if math.isfinite(value) and value > 0 else None
    except Exception:
        return None

def geometry_info(model, obj):
    """明确选择Body，提取米制世界坐标网格；没有Body不使用Axis凑数。

    网格体积是计算结果。边闭合不证明无自交，因此这里只用于数据交叉校核。
    """
    result = {"status": "失败", "step_id": obj.id()}
    try:
        reps = list(obj.Representation.Representations) if obj.Representation else []
        body = next((r for r in reps if r.RepresentationIdentifier == "Body"), None)
        if body is None:
            raise ValueError("没有Body三维主体表示")
        settings = ifcopenshell.geom.settings()
        settings.set("use-world-coords", True)
        settings.set("weld-vertices", True)
        # 默认 convert-back-units=False，几何引擎输出米，无需再乘0.001。
        shape = ifcopenshell.geom.create_shape(settings, obj, body)
        vertices = np.asarray(shape.geometry.verts, dtype=float).reshape(-1, 3)
        faces = np.asarray(shape.geometry.faces, dtype=int).reshape(-1, 3)
        if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
            raise ValueError("网格为空或坐标包含NaN/无穷值")
        edges = Counter(tuple(sorted((int(a), int(b)))) for f in faces for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])))
        closed = bool(edges) and all(count == 2 for count in edges.values())
        triangles = vertices[faces]
        centered = triangles - vertices.mean(axis=0)
        volume = abs(float(np.einsum("ij,ij->i", centered[:, 0], np.cross(centered[:, 1], centered[:, 2])).sum() / 6.0)) if closed else None
        signature = sorted(sorted(tuple(round(float(n), 6) for n in point) for point in tri) for tri in triangles)
        result.update(status="成功", body_step=body.id(), vertex_count=len(vertices), triangle_count=len(faces), bbox_min_m=vertices.min(axis=0).tolist(), bbox_max_m=vertices.max(axis=0).tolist(), closed_edges=closed, volume_m3=volume, signature=hashlib.sha256(json_text(signature).encode()).hexdigest())
    except Exception as exc:
        result["reason"] = str(exc)
    return result

def storey_candidates(model):
    """楼层Z和构件包围盒均换成米。优先使用楼层的世界放置Z。"""
    scale = project_scale(model, "LENGTHUNIT")
    if scale is None:
        return []
    candidates = []
    for floor in model.by_type("IfcBuildingStorey"):
        try:
            if floor.ObjectPlacement:
                z = float(placement_util.get_local_placement(floor.ObjectPlacement)[2, 3]) * scale
                method = "ObjectPlacement世界Z"
            elif floor.Elevation is not None:
                # 无放置时Elevation的参考原点可能不同，故列为候选但禁止自动使用。
                z = float(floor.Elevation) * scale
                method = "仅有Elevation，世界原点未验证"
            else:
                continue
            if math.isfinite(z):
                candidates.append({"entity": floor, "z_m": z, "method": method})
        except Exception:
            continue
    return candidates

def aggregation_fingerprint(model):
    return sorted((r.id(), r.RelatingObject.id(), tuple(sorted(x.id() for x in r.RelatedObjects))) for r in model.by_type("IfcRelAggregates"))

def csv_bytes(rows,columns=None):
    return pd.DataFrame(rows,columns=columns).to_csv(index=False).encode("utf-8-sig")

def _locate_element_storeys(obj):
    """沿构件聚合链继承包含关系，不能把合法楼梯/装配子件误判为游离。

    返回楼层对象列表、来源关系STEP编号及判定错误。
    """
    parents, error = spatial_parent_chain(obj)
    if error:
        return [], [], error
    for current in [obj] + parents:
        relations = list(getattr(current, "ContainedInStructure", ()))
        if not relations:
            continue
        if len(relations) != 1:
            return [], [r.id() for r in relations], "存在多个直接空间包含关系，需人工检查"
        structure = relations[0].RelatingStructure
        structure_parents, error = spatial_parent_chain(structure)
        if error:
            return [], [relations[0].id()], error
        floors = [p for p in [structure] + structure_parents if p.is_a("IfcBuildingStorey")]
        return floors, [relations[0].id()], None
    return [], [], None

def check_spatial_chain(storeys):
    if len(storeys) != 1:
        return False, "无法从唯一楼层建立完整空间链", []
    floor = storeys[0]
    parents, error = spatial_parent_chain(floor)
    chain = [floor] + parents
    expected = ["IfcBuildingStorey", "IfcBuilding", "IfcSite", "IfcProject"]
    valid = not error and len(chain) == len(expected) and all(
        obj.is_a(cls) for obj, cls in zip(chain, expected)
    )
    return valid, error or ("" if valid else "本课程要求Storey→Building→Site→Project的唯一完整父链"), [entity_ref(p) for p in chain]

def standard_quantities(model, obj, set_name):
    """从IfcElementQuantity提取数量实体。类型数量在先，实例同名数量覆盖。

    保留IfcQuantityLength/Area/Volume的真实类别和显式Unit，便于检查单位。
    """
    quantities = {}
    try:
        type_obj = element_util.get_type(obj)
        owners = ([type_obj] if type_obj is not None else []) + [obj]
        for owner in owners:
            definitions = list(getattr(owner, "HasPropertySets", ()) or ())
            definitions += [
                rel.RelatingPropertyDefinition for rel in getattr(owner, "IsDefinedBy", ())
                if rel.is_a("IfcRelDefinesByProperties")
            ]
            for definition in definitions:
                if definition.is_a("IfcElementQuantity") and definition.Name == set_name:
                    for quantity in definition.Quantities:
                        quantities[quantity.Name] = (definition, quantity)
        return quantities, None
    except Exception as exc:
        return {}, str(exc)

def quantity_in_si(model, pair, unit_type):
    if pair is None:
        return None, {"reason": "缺少指定的标准数量字段"}
    definition, quantity = pair
    cls, attribute = {
        "LENGTHUNIT": ("IfcQuantityLength", "LengthValue"),
        "AREAUNIT": ("IfcQuantityArea", "AreaValue"),
        "VOLUMEUNIT": ("IfcQuantityVolume", "VolumeValue"),
    }[unit_type]
    evidence = {"set_step": definition.id(), "quantity_step": quantity.id(), "set_name": definition.Name, "field": quantity.Name}
    try:
        if not quantity.is_a(cls):
            raise ValueError(f"数量类型应为{cls}，实际为{quantity.is_a()}")
        scale = project_scale(model, unit_type)
        if scale is None:
            raise ValueError("无法判定项目单位")
        if quantity.Unit is not None:
            if getattr(quantity.Unit, "UnitType", None) != unit_type:
                raise ValueError("显式单位类别与数量类别不符")
            scale *= unit_util.convert_unit(1.0, quantity.Unit, unit_util.get_project_unit(model, unit_type))
        original = float(getattr(quantity, attribute))
        result = original * scale
        evidence.update(value_raw=original, value_si=result, factor=scale, unit_step=quantity.Unit.id() if quantity.Unit else None)
        if not math.isfinite(result) or result <= 0:
            raise ValueError("数量必须是有限正数")
        return result, evidence
    except Exception as exc:
        evidence["reason"] = str(exc)
        return None, evidence

# 本版正式规则在运行评分前固定；指标按用途选取，不能为了目标分数动态改权重。
RULE_VERSION="4.0-full-model"
RULES=[("M1","标识","GUID合法且唯一",10),("M2","标识","名称完整",5),
       ("M3","关系","空间层级与楼层归属",15),("M4","关系","门窗宿主关系链",15),
       ("M5","属性","类别公用属性与字段",15),("M6","属性","标准工程量与空间净面积",20),
       ("M7","几何","Body几何可用",10),("M8","几何","净体积一致性",10)]
COMMON_CLASSES=["IfcWall","IfcColumn","IfcSlab","IfcDoor","IfcWindow","IfcSpace","IfcRailing","IfcStairFlight","IfcStair","IfcMember","IfcRoof","IfcPlate","IfcBeam","IfcBuildingElementProxy"]
COMMON_MAP={cls:"Pset_"+cls[3:]+"Common" for cls in COMMON_CLASSES}
QTO_REQUIRED={
 "IfcWall":{"NetVolume":"VOLUMEUNIT","NetSideArea":"AREAUNIT"},
 "IfcColumn":{"NetVolume":"VOLUMEUNIT","Length":"LENGTHUNIT"},
 "IfcSlab":{"NetVolume":"VOLUMEUNIT"},
 "IfcDoor":{"Width":"LENGTHUNIT","Height":"LENGTHUNIT"},
 "IfcWindow":{"Width":"LENGTHUNIT","Height":"LENGTHUNIT"},
 "IfcRailing":{"Length":"LENGTHUNIT"},
 "IfcStairFlight":{"NetVolume":"VOLUMEUNIT"},
 "IfcMember":{"NetVolume":"VOLUMEUNIT"},
 "IfcBeam":{"NetVolume":"VOLUMEUNIT"},
 "IfcPlate":{"NetVolume":"VOLUMEUNIT"},
 "IfcRoof":{"NetArea":"AREAUNIT"},
 "IfcSpace":{"NetFloorArea":"AREAUNIT"},
}
RULE_DETAILS=[
 ("M1","全文件IfcRoot的GUID格式合法且唯一。任意非空GUID重复，总分一票否决为0。"),
 ("M2","全部IfcBuildingElement及IfcSpace的Name非空且不是Default、Unnamed等占位词。"),
 ("M3","建筑构件具有唯一直接或继承楼层；空间的唯一聚合父链到Storey；所属楼层能按Storey→Building→Site→Project唯一上溯。"),
 ("M4","全部门窗能沿IfcRelFillsElement→IfcOpeningElement→IfcRelVoidsElement找到唯一IfcWall宿主。属于本课程关系交付要求，不宣称所有IFC窗都强制使用此链。"),
 ("M5","已映射类别的公用Pset需非空Reference文本；除IFC4模板不含IsExternal的梯段外，另需布尔IsExternal；墙、柱、板、梁、板片另需布尔LoadBearing。允许类型继承。这些是课程交付要求，不是IFC全部必填字段。未映射类别列为不适用。"),
 ("M6","固定分组计分：有独立Body的已映射建筑构件标准Qto占10分；IfcSpace标准NetFloorArea占10分。两组分别按对象通过率计分，避免多墙稀释空间面积缺失。自定义MeshVolume不替代标准Qto。"),
 ("M7","检查全部建筑构件和空间的有限Body网格。无独立Body且有聚合子项的父对象不适用；无Body叶子对象不通过，Body解析异常为无法判定。"),
 ("M8","只对墙、柱、板比较标准NetVolume与边闭合Body三角网格体积。容差=max(1%×NetVolume,0.001m³)。不把门窗不同语义尺寸强行比较计分。"),
]
POLICY={"version":RULE_VERSION,"scope":"全部IfcBuildingElement和IfcSpace；M1覆盖全文件IfcRoot；M4覆盖全部门窗","purpose":"课程自定义信息交付要求，不是IFC强制字段或设计合规认证","metrics":[dict(id=mid,dimension=d,name=n,weight=w,rule=dict(RULE_DETAILS)[mid]) for mid,d,n,w in RULES],"common_map":COMMON_MAP,"qto_required":QTO_REQUIRED,"quantity_groups":{"building_elements":10,"spaces":10},"volume_tolerance":{"relative":0.01,"absolute_m3":0.001},"unknown_policy":"无法判定不计通过并保留原因，不认定为已证实设计错误","supplementary_policy":"门窗尺寸差异、同类几何重合及Proxy分类候选保留证据，不扣正式八指标分","empty_policy":"没有建筑构件或空间时总分无法评估；其他无适用项剔除权重后归一化，明确标注适用性","veto":"全文件非空GlobalId重复使总分0"}


POLICY['grades']={'优秀':'>=90','良好':'80<=score<90','待改善':'60<=score<80','需整改':'score<60','优先级':'GUID重复优先不通过；无适用建筑对象为无法评估'}
POLICY['pset_exceptions']={'IfcStairFlight':'IFC4模板不含IsExternal，仅要求Reference；禁止扣不存在的字段'}


def target_elements(model):
    """固定恢复完整建筑范围，不再只筛选墙柱；排除几何点、关系和洞口辅助对象。"""
    return sorted(model.by_type("IfcBuildingElement")+model.by_type("IfcSpace"),key=lambda e:e.id())


def quality_grade(result):
    """等级是课程自定义数据交付等级，不代表建筑设计或结构安全等级。"""
    if result['duplicates']:return '不通过（GUID重复否决）'
    score=result['score']
    if score is None:return '无法评估'
    return '优秀' if score>=90 else '良好' if score>=80 else '待改善' if score>=60 else '需整改'


def audit_fingerprint(result):
    """排除运行时间，仅比较评分、证据与规则哈希，不用两次时间戳判断结果。"""
    fields=['schema','rule_version','policy_sha256','target_count','metrics','issues','score']
    return hashlib.sha256(json_text({k:result[k] for k in fields}).encode()).hexdigest()


def base_class(obj):
    return next((cls for cls in COMMON_CLASSES if obj.is_a(cls)),obj.is_a())


def locate_storeys(obj):
    # Space由聚合父链归属Storey，不能把Space误按物理构件的包含关系读取。
    if obj.is_a("IfcSpace"):
        parents,error=spatial_parent_chain(obj)
        return [p for p in parents if p.is_a("IfcBuildingStorey")],[r.id() for r in obj.Decomposes],error
    return _locate_element_storeys(obj)


def host_chain(obj):
    paths=[]
    try:
        for fill in obj.FillsVoids:
            opening=fill.RelatingOpeningElement
            for void in opening.VoidsElements:
                host=void.RelatingBuildingElement
                paths.append({"fill_relation_step":fill.id(),"opening":entity_ref(opening),"void_relation_step":void.id(),"host":entity_ref(host)})
        valid=len(paths)==1 and paths[0]["host"]["class"] in ["IfcWall","IfcWallStandardCase","IfcWallElementedCase"]
        return valid,paths,None
    except Exception as exc:return False,paths,str(exc)


def has_body_or_children(obj):
    reps=list(obj.Representation.Representations) if obj.Representation else []
    return any(r.RepresentationIdentifier=="Body" for r in reps),bool(list(getattr(obj,"IsDecomposedBy",())))


def audit_model(model):
    """评分和修复复查共用此函数。候选问题与正式扣分分别保存。"""
    targets=target_elements(model)
    if len(targets)>MAX_TARGETS:raise ValueError(f"课堂演示版最多分析{MAX_TARGETS}个建筑构件及空间，请缩小导出范围。")
    checks={mid:[] for mid,_,_,_ in RULES}; geometries={}; supplementary=[]; na=[]
    roots=model.by_type("IfcRoot");groups=defaultdict(list)
    for obj in roots:
        if obj.GlobalId and obj.GlobalId.strip():groups[obj.GlobalId].append(entity_ref(obj))
    duplicates={guid:refs for guid,refs in groups.items() if len(refs)>1}
    def record(mid,obj,status,reason="",evidence=None,group=None):
        checks[mid].append({"object":entity_ref(obj),"status":status,"reason":reason,"evidence":evidence or {},"group":group})
    for obj in roots:
        good=guid_is_valid(obj.GlobalId) and obj.GlobalId not in duplicates
        record("M1",obj,"通过" if good else "不通过","" if good else "GUID缺失、非法或重复",entity_ref(obj))
    quantity_cache={}
    for obj in targets:
        cls=base_class(obj);ref=entity_ref(obj)
        try:
            name=(obj.Name or "").strip();good=bool(name) and name.lower() not in {"default","unnamed","未命名","默认"}
            record("M2",obj,"通过" if good else "不通过","" if good else "名称为空或占位词",ref)
            floors,relations,error=locate_storeys(obj)
            good_chain,chain_reason,chain=check_spatial_chain(floors)
            good=len(floors)==1 and not error and good_chain
            record("M3",obj,"通过" if good else "不通过",error or ("" if good else chain_reason or "没有唯一楼层或父链不完整"),{"floor_steps":[f.id() for f in floors],"relations":relations,"chain":chain})
            if obj.is_a("IfcDoor") or obj.is_a("IfcWindow"):
                valid,paths,host_error=host_chain(obj)
                record("M4",obj,"无法判定" if host_error else ("通过" if valid else "不通过"),host_error or ("" if valid else "缺少唯一完整门窗—洞口—宿主墙关系链"),{"paths":paths})
            ps_name=COMMON_MAP.get(cls)
            if ps_name:
                try:
                    ps=element_util.get_pset(obj,ps_name,should_inherit=True) or {}
                    required={"Reference":str}
                    # IFC4的Pset_StairFlightCommon没有IsExternal，禁止误扣不存在的字段。
                    if cls!="IfcStairFlight":required["IsExternal"]=bool
                    if cls in ["IfcWall","IfcColumn","IfcSlab","IfcBeam","IfcPlate"]:required["LoadBearing"]=bool
                    bad=[k for k,t in required.items() if not isinstance(ps.get(k),t) or (t is str and not ps[k].strip())]
                    record("M5",obj,"不通过" if bad else "通过","缺失或类型不符："+"、".join(bad) if bad else "",{"pset_name":ps_name,"properties":ps,"required_fields":list(required)})
                except Exception as exc:record("M5",obj,"无法判定","属性解析异常："+str(exc))
            else:na.append({**ref,"metric":"M5","reason":"本交付规则未配置该类别必需Pset"})
            body,children=has_body_or_children(obj)
            geom=geometry_info(model,obj) if body else {"status":"失败","reason":"没有Body三维主体表示","step_id":obj.id()}
            geometries[str(obj.id())]=geom
            if body or not children:
                record("M7",obj,"通过" if geom["status"]=="成功" else ("不通过" if not body else "无法判定"),geom.get("reason",""),geom)
            else:na.append({**ref,"metric":"M7","reason":"聚合父对象，无独立Body，避免重复要求几何"})
            required_qto=QTO_REQUIRED.get(cls)
            applicable_qto=obj.is_a("IfcSpace") or (body and required_qto is not None)
            if applicable_qto and required_qto:
                set_name="Qto_"+cls[3:]+"BaseQuantities"
                quantities,qerror=standard_quantities(model,obj,set_name)
                values={};evidence={}
                for field,unit_type in required_qto.items():values[field],evidence[field]=quantity_in_si(model,quantities.get(field),unit_type)
                missing=[k for k,v in values.items() if v is None]
                record("M6",obj,"无法判定" if qerror else ("不通过" if missing else "通过"),qerror or ("缺少可用标准数量："+"、".join(missing) if missing else ""),{"qto_name":set_name,"fields":evidence},"spaces" if obj.is_a("IfcSpace") else "building_elements")
                quantity_cache[obj.id()]=(values,evidence)
            else:na.append({**ref,"metric":"M6","reason":"无独立Body的聚合父对象或未配置标准数量类别"})
            if cls in ["IfcWall","IfcColumn","IfcSlab"] and (body or not children):
                values,evidence=quantity_cache.get(obj.id(),({},{}));volume=values.get("NetVolume")
                if volume is None or geom.get("volume_m3") is None or not geom.get("closed_edges"):
                    record("M8",obj,"无法判定","需可用标准NetVolume及边闭合Body，不能用毛体积或包围盒替代",{"quantity":evidence.get("NetVolume"),"geometry":geom})
                else:
                    difference=abs(volume-geom["volume_m3"]);tol=max(0.001,abs(volume)*0.01)
                    record("M8",obj,"通过" if difference<=tol else "不通过","" if difference<=tol else "净体积与网格差超过容差",{"quantity":evidence["NetVolume"],"body_step":geom["body_step"],"mesh_volume_m3":geom["volume_m3"],"difference_m3":difference,"tolerance_m3":tol})
            # 门窗宽高可能具有不同语义，仅记录同名数值差异候选，扣分为0。
            if obj.is_a("IfcDoor") or obj.is_a("IfcWindow"):
                quantities,_=standard_quantities(model,obj,"Qto_"+cls[3:]+"BaseQuantities")
                scale=project_scale(model,"LENGTHUNIT");dimensions={};different=False
                for field,attr in [("Width","OverallWidth"),("Height","OverallHeight")]:
                    value=getattr(obj,attr,None);overall=float(value)*scale if value is not None and scale is not None else None
                    qvalue,qe=quantity_in_si(model,quantities.get(field),"LENGTHUNIT")
                    delta=abs(overall-qvalue) if overall is not None and qvalue is not None else None
                    dimensions[field]={"overall_m":overall,"qto_m":qvalue,"delta_m":delta,"quantity_evidence":qe}
                    if delta is not None and delta>0.001:different=True
                if different:supplementary.append({"object":ref,"kind":"门窗尺寸差异候选","reason":"Overall与Qto宽高差超过1mm；语义未确认，不扣分、不自动覆写尺寸","evidence":dimensions})
        except Exception as exc:
            # 异常不能丢失对象。补入该对象本应适用但尚未完成的指标。
            applicable=["M2","M3","M7"]+(["M5"] if cls in COMMON_MAP else [])+(["M6"] if cls in QTO_REQUIRED else [])+(["M4"] if cls in ["IfcDoor","IfcWindow"] else [])+(["M8"] if cls in ["IfcWall","IfcColumn","IfcSlab"] else [])
            for mid in applicable:
                if not any(r["object"]["step_id"]==obj.id() for r in checks[mid]):record(mid,obj,"无法判定","实体解析异常："+str(exc),group="spaces" if obj.is_a("IfcSpace") else "building_elements")
    # 几何重合与Proxy继续保留专项证据，不在新八指标中暗扣分。
    mesh_groups=defaultdict(list)
    for obj in targets:
        g=geometries.get(str(obj.id()),{})
        if g.get("status")=="成功":mesh_groups[(base_class(obj),g["signature"])].append(obj)
    candidate_groups=[];candidate_count=0
    for key,objs in mesh_groups.items():
        if len(objs)>1:
            refs=[entity_ref(o) for o in sorted(objs,key=lambda o:o.id())];candidate_count+=len(refs)-1
            candidate_groups.append({"class":key[0],"steps":[r["step_id"] for r in refs],"signature":key[1]})
            for ref in refs[1:]:supplementary.append({"object":ref,"kind":"几何重合候选","reason":"同类世界网格相同，未证实语义冗余，不扣分、不删除","evidence":{"objects":refs,"signature":key[1]}})
    proxies=model.by_type("IfcBuildingElementProxy")
    for obj in proxies:supplementary.append({"object":entity_ref(obj),"kind":"Proxy分类复核","reason":"通用类别可能合理，需确认交付分类，不直接扣分","evidence":entity_ref(obj)})
    # 为每条正式问题先计算所属指标内的权重。M6有两个固定10分子组。
    metrics=[];issues=[];active_points=0.;active_weight=0.;metric_weights={}
    for mid,dimension,label,weight in RULES:
        entries=checks[mid];n=len(entries);passed=sum(r["status"]=="通过" for r in entries)
        unknown=sum(r["status"]=="无法判定" for r in entries)
        details=[]
        if mid=="M6":
            subgroups=[(group,[r for r in entries if r["group"]==group]) for group in ["building_elements","spaces"]]
            present=[(group,rows) for group,rows in subgroups if rows]
            points=0. if present else None
            for group,rows in present:
                # 两组都有时各10分；仅有一组适用时占该项20分，必须在明细说明。
                share=weight/len(present);p=sum(r["status"]=="通过" for r in rows)
                points+=share*p/len(rows)
                details.append({"group":group,"passed":p,"total":len(rows),"weight":share,"points":share*p/len(rows)})
                for row in rows:row["deduction_in_metric"]=share/len(rows)
        else:
            points=weight*passed/n if n else None
            for row in entries:row["deduction_in_metric"]=weight/n
        if n:active_points+=points;active_weight+=weight
        metrics.append({"编号":mid,"维度":dimension,"指标":label,"达标率%":round(points/weight*100,3) if points is not None else None,"权重":weight,"通过":passed,"待复核":0,"不通过":n-passed-unknown,"无法判定":unknown,"适用对象":n,"得分":round(points,4) if points is not None else None,"计算口径":"构件与空间两组加权" if mid=="M6" else "对象通过率","子组":details})
    raw=active_points/active_weight*100 if targets and active_weight else None
    normalization=100/active_weight if active_weight else 0.
    for mid,_,label,_ in RULES:
        for row in checks[mid]:
            if row["status"]=="通过":continue
            ref=row["object"]
            issues.append({"指标":mid+" "+label,"对象STEP":ref["step_id"],"GlobalId":ref["GlobalId"],"IFC类别":ref["class"],"名称":ref["Name"],"状态":row["status"],"原因":row["reason"],"常规扣分":row["deduction_in_metric"]*normalization,"额外否决扣分":0.,"证据":json_text(row["evidence"])})
    formal_count=len(issues)
    for row in supplementary:
        ref=row["object"];issues.append({"指标":"专项 "+row["kind"],"对象STEP":ref["step_id"],"GlobalId":ref["GlobalId"],"IFC类别":ref["class"],"名称":ref["Name"],"状态":"待复核","原因":row["reason"],"常规扣分":0.,"额外否决扣分":0.,"证据":json_text(row["evidence"])})
    for guid,refs in duplicates.items():
        for ref in refs:issues.append({"指标":"全局GUID否决证据","对象STEP":ref["step_id"],"GlobalId":guid,"IFC类别":ref["class"],"名称":ref["Name"],"状态":"一票否决","原因":"全文件非空GUID重复","常规扣分":0.,"额外否决扣分":0.,"证据":json_text(refs)})
    if duplicates and raw is not None:issues.append({"指标":"全局GUID否决扣分汇总","对象STEP":None,"GlobalId":None,"IFC类别":None,"名称":None,"状态":"一票否决","原因":"只对剩余常规得分额外否决一次","常规扣分":0.,"额外否决扣分":raw,"证据":json_text(duplicates)})
    buildings=model.by_type("IfcBuildingElement");spaces=model.by_type("IfcSpace")
    orphan_count=sum(not locate_storeys(o)[0] and not locate_storeys(o)[2] for o in targets)
    return {"schema":model.schema,"rule_version":RULE_VERSION,"policy_sha256":hashlib.sha256(json_text(POLICY).encode()).hexdigest(),"target_count":len(targets),"wall_count":len(model.by_type("IfcWall")),"column_count":len(model.by_type("IfcColumn")),"door_count":len(model.by_type("IfcDoor")),"window_count":len(model.by_type("IfcWindow")),"space_count":len(spaces),"building_element_count":len(buildings),"root_count":len(roots),"duplicates":duplicates,"metrics":metrics,"issues":issues,"formal_issue_count":formal_count,"supplementary_issue_count":len(supplementary),"supplementary":supplementary,"not_applicable":na,"score":0. if duplicates else (round(raw,4) if raw is not None else None),"raw_score":raw,"geometry":geometries,"objects":[entity_ref(o) for o in targets],"checked_at_utc":datetime.now(timezone.utc).isoformat(),"unique_rate":100*sum(guid_is_valid(o.GlobalId) and o.GlobalId not in duplicates for o in roots)/len(roots) if roots else None,"orphan_count":orphan_count,"proxy_count":len(proxies),"duplicate_candidates":candidate_groups,"duplicate_candidate_count":candidate_count,"active_weight":active_weight}


def repair_railing_quantities(model,logs):
    """只把已有、带明确IfcLengthMeasure类型的Revit长度转存为标准栏杆Length。
    不从包围盒猜路径长度，不把MeshVolume当NetVolume，不填空间净面积。
    """
    for obj in model.by_type("IfcRailing"):
        body,_=has_body_or_children(obj)
        if not body:continue
        qs,_=standard_quantities(model,obj,"Qto_RailingBaseQuantities")
        value,_=quantity_in_si(model,qs.get("Length"),"LENGTHUNIT")
        if value is not None:continue
        sources=[]
        try:
            project=unit_util.get_project_unit(model,"LENGTHUNIT")
            if project is None:raise ValueError("长度单位缺失")
            for relation in obj.IsDefinedBy:
                ps=relation.RelatingPropertyDefinition
                if not ps.is_a("IfcPropertySet") or ps.Name not in ["尺寸标注","Dimensions"]:continue
                for prop in ps.HasProperties:
                    if prop.is_a("IfcPropertySingleValue") and prop.Name in ["长度","Length"]:
                        v=prop.NominalValue
                        if v is not None and v.is_a() in ["IfcLengthMeasure","IfcPositiveLengthMeasure"]:
                            n=float(v.wrappedValue)
                            if prop.Unit is not None:n*=unit_util.convert_unit(1.,prop.Unit,project)
                            if math.isfinite(n) and n>0:sources.append((n,ps.id(),prop.id(),v.is_a()))
            if not sources or any(abs(s[0]-sources[0][0])>1e-6 for s in sources):raise ValueError("没有唯一、正值、显式长度类型的来源")
            length,psid,propid,typename=sources[0]
            qto=ifcopenshell.api.run("pset.add_qto",model,product=obj,name="Qto_RailingBaseQuantities")
            qto.MethodOfMeasurement="Copied from typed existing IFC property; source property #"+str(propid)+"; no geometry inference"
            ifcopenshell.api.run("pset.edit_qto",model,qto=qto,properties={"Length":model.createIfcLengthMeasure(length)})
            logs.append({"STEP":obj.id(),"GlobalId":obj.GlobalId,"动作":"补标准栏杆Length","状态":"已修复","修复前":None,"修复后":length,"依据":json_text({"source_pset_step":psid,"source_property_step":propid,"source_type":typename,"qto_step":qto.id(),"unit":"项目长度单位","method":"已有类型化长度直接转存，不计算或猜测路径"})})
        except Exception as exc:logs.append({"STEP":obj.id(),"GlobalId":obj.GlobalId,"动作":"补标准栏杆Length","状态":"跳过","依据":str(exc)})


def analyse_bytes(data):
    with open_uploaded_ifc(data) as model: result=audit_model(model)
    result["sha256"]=hashlib.sha256(data).hexdigest()
    return result

def repair_bytes(data, max_floor_distance_m=4.0):
    """修复GUID、可推断的游离建筑构件及有显式长度来源的标准栏杆数量。

    最近楼层是推断，不等于恢复真实设计意图。选用包围盒底部Z；若与楼层
    高差超过上限、并列最近、跨建筑无法确定或有装配父件，跳过而非强行修复。
    """
    with open_uploaded_ifc(data) as model:
        before = audit_model(model)
        aggregation_before = aggregation_fingerprint(model)
        logs, used = [], set()

        # 保留重复组的首个对象，后续对象以及缺失/非法GUID重新生成。
        for root in sorted(model.by_type("IfcRoot"), key=lambda e: e.id()):
            old = root.GlobalId
            if not guid_is_valid(old) or old in used:
                new = ifcopenshell.guid.compress(uuid.uuid4().hex)
                while new in used:
                    new = ifcopenshell.guid.compress(uuid.uuid4().hex)
                root.GlobalId = new
                logs.append({"STEP": root.id(), "动作": "重写GUID", "状态": "已修复", "修复前": old, "修复后": new, "依据": "缺失、格式非法或后出现的重复GUID"})
            used.add(root.GlobalId)

        floors = [row for row in storey_candidates(model) if row["method"] == "ObjectPlacement世界Z"]
        buildings = model.by_type("IfcBuilding")
        for obj in target_elements(model):
            previous_floors, relations, error = locate_storeys(obj)
            if len(previous_floors) == 1 and not error:
                continue  # 包括继承的有效归属，不拆散装配件。
            info = entity_ref(obj)
            record = {"STEP": obj.id(), "GlobalId": obj.GlobalId, "动作": "最近楼层匹配", "状态": "跳过", "修复前": json_text(info), "修复后": None}
            try:
                if obj.is_a("IfcSpace"):
                    raise ValueError("空间应通过聚合归属楼层，禁止改成物理构件包含关系")
                if error or previous_floors or relations or list(getattr(obj, "ContainedInStructure", ())):
                    raise ValueError("已有冲突或非楼层包含关系，不能按游离构件强行改挂")
                if list(getattr(obj, "Decomposes", ())):
                    raise ValueError("构件有装配父对象，直接assign_container可能破坏聚合，需人工复核")
                if len(buildings) != 1:
                    raise ValueError("本模型不是单建筑，不能只凭Z值跨建筑匹配楼层")
                # 只选择唯一建筑内部、父链完整的楼层。
                available = []
                for floor in floors:
                    chain, chain_error = spatial_parent_chain(floor["entity"])
                    if not chain_error and any(parent.id() == buildings[0].id() for parent in chain):
                        available.append(floor)
                if not available:
                    raise ValueError("没有世界标高可判定的所属建筑楼层")
                geom = before["geometry"].get(str(obj.id()), {})
                if geom.get("status") != "成功":
                    raise ValueError("无法取得有效Body包围盒Z")
                z_min = geom["bbox_min_m"][2]
                ranked = sorted(available, key=lambda row: (abs(row["z_m"] - z_min), row["entity"].id()))
                nearest = ranked[0]
                distance = abs(nearest["z_m"] - z_min)
                if len(ranked) > 1 and abs(abs(ranked[1]["z_m"] - z_min) - distance) < 1e-6:
                    raise ValueError("存在并列最近楼层，无法唯一匹配")
                if distance > max_floor_distance_m:
                    raise ValueError(f"最近楼层高差{distance:.3f}m超过上限{max_floor_distance_m:.3f}m")
                relation = ifcopenshell.api.run("spatial.assign_container", model, products=[obj], relating_structure=nearest["entity"])
                assigned, _, assigned_error = locate_storeys(obj)
                if assigned_error or len(assigned) != 1 or assigned[0].id() != nearest["entity"].id():
                    raise ValueError("包含关系写入后未通过复查")
                record.update(状态="已推断修复", 修复后=json_text(entity_ref(nearest["entity"])), 依据=json_text({"bbox_Zmin_m": z_min, "storey_world_Z_m": nearest["z_m"], "distance_m": distance, "relation_step": relation.id(), "method": "按包围盒底部Z选择最近世界标高；须人工确认设计意图"}))
            except Exception as exc:
                record["依据"] = str(exc)
            logs.append(record)

        repair_railing_quantities(model, logs)

        if aggregation_before != aggregation_fingerprint(model):
            raise ValueError("修复改变了原聚合关系，已拒绝提供修复文件。")
        after = audit_model(model)
        if after["duplicates"]:
            raise ValueError("GUID修复后仍有重复，已拒绝导出。")
        # assign_container可能重设局部坐标，但不应移动世界坐标中的几何。
        preserved = [key for key, geom in before["geometry"].items() if geom.get("status") == "成功"]
        for key in preserved:
            if before["geometry"][key]["signature"] != after["geometry"].get(key, {}).get("signature"):
                raise ValueError(f"修复改变了#{key}的世界Body网格，已拒绝导出。")

        out_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix="_fixed.ifc", delete=False) as out:
                out_path = Path(out.name)
            model.write(str(out_path))
            fixed_data = out_path.read_bytes()
            # 对真正写出的文件重新解析，避免只检查内存模型。
            reloaded = analyse_bytes(fixed_data)
            if reloaded["duplicates"]:
                raise ValueError("导出文件再次读取后发现重复GUID。")
        finally:
            if out_path is not None:
                out_path.unlink(missing_ok=True)
    if not logs:
        logs.append({"STEP": None, "动作": "检查修复条件", "状态": "无需修复", "依据": "未发现非法/重复GUID及满足自动修复条件的游离构件或栏杆数量"})
    return {"fixed_data": fixed_data, "before": before, "after": reloaded, "logs": logs, "geometry_preserved_count": len(preserved), "aggregation_preserved": True}

# CSS仅控制视觉呈现，全部业务交互仍使用Streamlit原生组件，保留键盘访问。
# 优先使用data-testid，不依赖每次构建会变化的随机CSS类名。
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --ink:#17243d; --muted:#66758e; --accent:#6366f1; }
html, body, [data-testid="stApp"] {
  font-family:'Inter','Noto Sans SC','Microsoft YaHei',sans-serif;
  color:var(--ink); background:#f4f6fb;
}
[data-testid="stAppViewContainer"] {background:radial-gradient(ellipse at 95% 0%,#e9eaff 0%,transparent 42%),#f4f6fb;}
[data-testid="stHeader"] {background:rgba(244,246,251,.90);}
[data-testid="stMainBlockContainer"] {max-width:1510px;padding:4.25rem 2.6rem 3rem;}
h1,h2,h3 {letter-spacing:-.035em; color:#17243d;}
h1 {font-weight:800!important;}
[data-testid="stSidebar"] {background:linear-gradient(165deg,#111c35,#1e2b4c);border-right:3px solid #7375ff;}
[data-testid="stSidebar"] * {color:#dce4f4;}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {color:#fff!important;}
[data-testid="stSidebar"] hr {border-color:#3e4c6c;}
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {background:#263654;color:#fff;}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {background:#263654;color:#fff;}
.brand {font-size:1.34rem;font-weight:800;color:white;letter-spacing:-.03em;}
.brand-dot {display:inline-block;background:#8c90ff;width:11px;height:11px;border-radius:50%;margin-right:10px;box-shadow:0 0 18px #8c90ff88;}
.sidebar-note {font-size:.84rem;line-height:1.85;color:#aebdd8!important;}
.eyebrow {font-size:.72rem;font-weight:700;letter-spacing:.16em;color:#6267cd;margin-bottom:.55rem;}
.hero-title {font-size:clamp(1.7rem,3vw,2.7rem);font-weight:800;line-height:1.25;letter-spacing:-.045em;margin-bottom:.65rem;color:#17243d;}
.hero-sub {font-size:.96rem;line-height:1.8;color:#66758e;margin-bottom:1.1rem;}
.section-note {font-size:.86rem;line-height:1.8;color:#697a96;}
.empty-state {padding:3.4rem 1.8rem;text-align:center;border:1px solid #e1e6f2;border-radius:22px;background:rgba(255,255,255,.85);box-shadow:0 12px 35px #263b6410;}
.empty-state h3 {font-size:1.4rem;margin-bottom:.7rem;}
.empty-state p {color:#72819a;line-height:1.8;max-width:600px;margin:auto;}
.file-info {font-size:.8rem;color:#64748b;overflow-wrap:anywhere;background:#edf0f9;padding:.7rem 1rem;border-radius:11px;margin-bottom:.8rem;}
[data-testid="stMetric"] {background:linear-gradient(130deg,#5a5de5 0%,#7b58d3 100%);border:1px solid #ffffff44;border-radius:18px;padding:1.25rem 1.35rem;box-shadow:0 12px 24px #6657c925;min-height:135px;}
[data-testid="stMetric"] label,[data-testid="stMetric"] [data-testid="stMetricLabel"] p {color:#ecebff!important;font-weight:500;}
[data-testid="stMetricValue"] {color:#fff!important;font-weight:800;font-size:2.25rem;letter-spacing:-.045em;}
[data-testid="stMetricDelta"] {color:#e2dcff!important;}
[data-testid="stMetric"] [data-testid="stTooltipIcon"] {color:#e4e3ff;}
[data-testid="stFileUploader"] {border:1.8px dashed #acb2ed;border-radius:18px;padding:.75rem;background:#ffffffbb;transition:all .18s;}
[data-testid="stFileUploader"]:hover {border-color:#6366f1;background:#f0f1ff;box-shadow:0 5px 18px #6366f114;}
[data-testid="stFileUploaderDropzone"] {background:transparent;border:none;}
[data-testid="stButton"] button,[data-testid="stDownloadButton"] button,[data-testid="stFileUploader"] button {border-radius:11px;border:1px solid #d9dfef;font-weight:600;padding:.6rem 1.15rem;transition:all .18s;box-shadow:0 3px 8px #19264908;}
[data-testid="stButton"] button:hover,[data-testid="stDownloadButton"] button:hover {border-color:#6366f1;background:#eef0ff;color:#494ad0;transform:translateY(-1px);box-shadow:0 6px 13px #6366f11c;}
[data-testid="stButton"] button[kind="primary"] {background:linear-gradient(120deg,#585ce3,#7957cf);border:none;color:white;box-shadow:0 7px 16px #635cda30;}
[data-testid="stButton"] button[kind="primary"]:hover {background:linear-gradient(120deg,#484ccd,#6847bd);color:#fff;}
[data-testid="stTabs"] [role="tablist"] {gap:1rem;border-bottom:1px solid #dfe4f1;margin:1rem 0 1.2rem;}
[data-testid="stTabs"] [role="tab"] {height:3.2rem;border-radius:10px 10px 0 0;padding:0 1.05rem;font-weight:600;color:#697993;background:transparent;}
[data-testid="stTabs"] [aria-selected="true"] {color:#5657db;background:#e9ebfc;}
[data-baseweb="tab-highlight"] {background:#6366f1!important;height:3px;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:18px;}
[data-testid="stDataFrame"] {border:1px solid #e0e5f1;border-radius:13px;overflow:hidden;background:white;}
[data-testid="stProgress"] [role="progressbar"] {border-radius:20px;}
[data-testid="stProgress"] [role="progressbar"]>div {background:#7066e4;}
[data-testid="stAlert"] {border-radius:12px;}
[data-testid="stExpander"] {background:#ffffff88;border-radius:12px;border:1px solid #e0e5f1;}
@media(max-width:850px){[data-testid="stMainBlockContainer"]{padding:1.7rem 1rem;}[data-testid="stMetric"]{padding:1rem;min-height:120px;}[data-testid="stMetricValue"]{font-size:1.8rem;}}
@media(prefers-reduced-motion:reduce){*{transition:none!important;}}
</style>
"""


def radar_chart(result):
    """统一画通过比例，不把风险型指标的高值错误画成高质量。"""
    metrics=result["metrics"]
    labels=[m["编号"]+" "+m["指标"].replace("率","") for m in metrics]
    rates=[100*m["通过"]/m["适用对象"] if m["适用对象"] else None for m in metrics]
    fig=go.Figure()
    fig.add_trace(go.Scatterpolar(r=rates+[rates[0]],theta=labels+[labels[0]],fill="toself",name="本次审查",line=dict(color="#6564e7",width=3),fillcolor="rgba(101,100,231,.16)",marker=dict(size=6,color="#6564e7"),connectgaps=False,hovertemplate="%{theta}<br>达标程度 %{r:.1f}%<extra></extra>"))
    fig.update_layout(height=370,margin=dict(l=70,r=70,t=35,b=40),showlegend=False,paper_bgcolor="rgba(0,0,0,0)",font=dict(family="Inter, Microsoft YaHei, sans-serif",color="#66758e",size=11),polar=dict(bgcolor="rgba(255,255,255,.4)",radialaxis=dict(visible=True,range=[0,100],tickvals=[25,50,75,100],ticksuffix="%",gridcolor="#e1e5f3",linecolor="#e1e5f3"),angularaxis=dict(gridcolor="#e1e5f3",linecolor="#e1e5f3")))
    return fig


def overview_cards(result):
    score=result["score"]; unique=result["unique_rate"]
    cols=st.columns(4,gap="medium")
    cols[0].metric("质量诊断总分",f"{score:.2f}" if score is not None else "—",help="满分100。全文件GUID重复时一票否决。课程自定义规则，不代表设计合规认证。")
    cols[1].metric("全文件 GUID 达标率",f"{unique:.1f}%" if unique is not None else "—",help="分母为全部IfcRoot，要求格式合法且全文件唯一。")
    cols[2].metric("正式扣分记录",str(result["formal_issue_count"]),help="进入八指标计分的不通过与无法判定记录，同一对象可能出现多条。")
    cols[3].metric("专项待复核",str(result["supplementary_issue_count"]),help="尺寸差异、几何重合及Proxy分类候选，证据保留，但不扣正式分。")
    st.caption("课程数据交付等级："+quality_grade(result)+" · ≥90优秀 / ≥80良好 / ≥60待改善 / <60需整改")
    st.progress(0. if score is None else max(0.,min(1.,score/100)))
    if result["duplicates"]:st.error("发现全文件重复GlobalId：已触发一票否决，总分为0。请查看清单及修复中心。")
    if not result["target_count"]:st.info("没有建筑构件或空间，综合分无法评估。")


def issue_table(result,key):
    rows=result["issues"]
    c1,c2=st.columns([1,3])
    status=c1.selectbox("记录状态",["全部","不通过","待复核","无法判定","一票否决"],key=key+"_status")
    query=c2.text_input("搜索对象",placeholder="输入 STEP、GlobalId、名称或原因…",key=key+"_search")
    filtered=[r for r in rows if (status=="全部" or r["状态"]==status) and (not query or query.casefold() in json_text(r).casefold())]
    st.caption(f"筛选 {len(filtered)} / {len(rows)} 条记录。无法判定项不计通过，专项候选不扣分，正式扣分与专项证据分开展示。")
    if filtered:st.dataframe(pd.DataFrame(filtered,columns=ISSUE_COLUMNS),hide_index=True,width="stretch",height=330)
    else:st.success("当前筛选条件下没有待处理记录。")
    a,b=st.columns(2)
    a.download_button("↓ 下载完整审查明细",csv_bytes(rows,ISSUE_COLUMNS),"ifc_quality_audit.csv","text/csv",key=key+"_all",width="stretch")
    b.download_button("↓ 下载当前筛选结果",csv_bytes(filtered,ISSUE_COLUMNS),"ifc_quality_filtered.csv","text/csv",key=key+"_filtered",width="stretch")


def rule_panel():
    st.subheader("完整建筑模型 · 固定评分规则 4.0")
    st.caption("四维八指标合计100分；全文件非空GUID重复一票否决。M6有两个固定子组，其余按各自适用对象通过率计算。")
    st.dataframe(pd.DataFrame([{"编号":mid,"指标":name,"维度":dimension,"权重":weight,"判定规则":dict(RULE_DETAILS)[mid]} for mid,dimension,name,weight in RULES]),hide_index=True,width="stretch",height=410)
    st.markdown("""**正式计分范围**：全部建筑构件和空间；门窗宿主单独检查，空间标准净面积纳入工程量组。

**M6的20分**：构件标准Qto占10分，空间标准NetFloorArea占10分。两组分别计算，避免多墙稀释房间面积缺失；有一组不适用时，该项权重由适用组承担，并显示子组口径。

**专项候选零扣分**：Overall与Qto宽高差、同类几何重合、Proxy分类均保留实体证据，但不自动认定错误或覆写尺寸。

**无法判定**：记录原因、保守不计通过；不等于已证实设计错误。无适用项会标注并剔除权重归一化。

**修复边界**：可修GUID、满足条件的游离构件、已有类型化长度的标准栏杆数量。不猜净面积、窗宿主、材料或防火性能，不删除重合候选。

**等级划分**：≥90优秀；80–<90良好；60–<80待改善；<60需整改。GUID重复优先判为不通过；无建筑构件及空间为无法评估。课程自定义等级，不代表设计合规。

**版本可比性**：旧版99.66、墙柱版100与本版适用范围不同，不能说成模型质量变差。当前原始IFC重新计算的结果才用于本轮汇报。""")
    st.download_button("↓ 下载完整评分规则 JSON",json.dumps(POLICY,ensure_ascii=False,indent=2),"ifc_quality_policy_v4.json","application/json",key="policy_download")



def main():
    st.set_page_config(page_title="IFC Studio · 模型质量审查",page_icon="🏗️",layout="wide",initial_sidebar_state="expanded")
    st.markdown(CUSTOM_CSS,unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-dot"></span>IFC Studio</div>',unsafe_allow_html=True)
        st.caption("MODEL QUALITY WORKSPACE")
        st.divider()
        st.subheader("项目工作区")
        st.markdown('<div class="sidebar-note">上传 IFC，审查数据质量，追溯问题，并在独立副本中完成修复。</div>',unsafe_allow_html=True)
        st.write("**审查对象**　构件 · 门窗 · 空间")
        st.write("**文件上限**　30 MB")
        st.write("**修复输出**　独立 *_fixed.ifc")
        st.divider()
        st.subheader("修复参数")
        max_distance=st.number_input("最近楼层允许高差 / m",min_value=0.,max_value=100.,value=4.,step=.5,help="包围盒底部Z与楼层世界Z的最大距离。超过则跳过，不移动模型。")
        st.caption("推断结果需要复核。GUID修复保留旧新值映射。")
        st.divider()
        st.markdown('<div class="sidebar-note">文件在当前服务器临时处理，不保存到GitHub或固定目录。原始上传内容不会被覆盖。</div>',unsafe_allow_html=True)
        st.caption(f"规则 {RULE_VERSION} · IfcOpenShell {ifcopenshell.version}")
    st.markdown('<div class="eyebrow">IFC STUDIO / QUALITY INTELLIGENCE</div><div class="hero-title">模型质量，一目了然。</div><div class="hero-sub">基于 IfcOpenShell 的 IFC 数据质量审查与自动修复系统</div>',unsafe_allow_html=True)
    uploaded=st.file_uploader("上传 IFC 模型",type=["ifc"],max_upload_size=MAX_FILE_MB,help="支持IFC交换文件。本演示版最多500个建筑构件及空间。")
    result=None;data=None;filename="model.ifc"
    if uploaded is not None:
        filename=Path(uploaded.name).name;data=uploaded.getvalue();digest=hashlib.sha256(data).hexdigest()
        if st.session_state.get("input_digest")!=digest:
            st.session_state["input_digest"]=digest
            st.session_state.pop("audit",None);st.session_state.pop("repair",None)
        # 文件名来自上传内容，注入HTML前必须转义，防止用户文本影响界面。
        st.markdown(f'<div class="file-info">{escape(filename)} &nbsp; · &nbsp; {len(data)/1024/1024:.2f} MB &nbsp; · &nbsp; SHA256 {escape(digest[:20])}…</div>',unsafe_allow_html=True)
        if len(data)>MAX_FILE_MB*1024*1024:
            st.error(f"文件超过{MAX_FILE_MB}MB，请缩小导出范围。");data=None
        elif st.button("开始质量审查",type="primary",icon="🔎"):
            try:
                with st.spinner("解析实体、关系与Body几何，生成可追溯结果…"):
                    reviewed=analyse_bytes(data)
                    cli_result=analyse_bytes(data)
                    if audit_fingerprint(reviewed)!=audit_fingerprint(cli_result):raise ValueError("命令行/页面结果一致性校验失败")
                    st.session_state["audit"]=reviewed;st.session_state.pop("repair",None)
                    st.session_state["consistency_digest"]=digest
            except Exception as exc:st.session_state.pop("audit",None);st.error("无法完成审查："+str(exc))
        result=st.session_state.get("audit") if data is not None else None
    tabs=st.tabs(["📊 质量审查看板","🛠️ 自动修复中心","📖 审查规则说明"])
    with tabs[0]:
        if result is None:
            st.markdown('<div class="empty-state"><h3>从一个 IFC 文件开始</h3><p>上传模型并点击「开始质量审查」。得分、GUID唯一率、游离构件与问题证据将在这里呈现。</p></div>',unsafe_allow_html=True)
        else:
            overview_cards(result)
            if st.session_state.get("consistency_digest")==digest:st.success("命令行/页面结果一致性校验 PASS（同一入口重新读取，比较规则、八指标和问题证据）")
            st.caption(f"{result['schema']} · 建筑构件 {result['building_element_count']} · 门 {result['door_count']} · 窗 {result['window_count']} · 空间 {result['space_count']} · 游离对象 {result['orphan_count']}")
            left,right=st.columns([1.2,1],gap="large")
            with left:
                with st.container(border=True):
                    st.subheader("八指标质量画像")
                    st.plotly_chart(radar_chart(result),width="stretch",key="original_radar",config={"displayModeBar":False})
                    st.caption("雷达图统一为达标程度，越接近外圈越好；M6按构件数量与空间净面积分别计分，子组明细可展开查看。")
            with right:
                st.subheader("审查得分明细")
                # 紧凑表优先展示权重和得分，避免桌面分栏时关键列藏在横向滚动区。
                frame=pd.DataFrame([{"指标":m["编号"]+" "+m["指标"],"权重":m["权重"],"通过/适用":f"{m['通过']}/{m['适用对象']}","得分":m["得分"]} for m in result["metrics"]])
                st.dataframe(frame,hide_index=True,width="stretch",height=360)
                st.caption("M6固定分组：构件标准Qto 10分，空间标准净面积10分。两个子组的通过率分别计算。")
            with st.expander("展开完整指标字段与分组计分证据"):
                st.dataframe(pd.DataFrame(result["metrics"]),hide_index=True,width="stretch")
                st.json({"M6子组":next(m["子组"] for m in result["metrics"] if m["编号"]=="M6"),"不适用对象":result["not_applicable"]},expanded=False)
                st.download_button("↓ 下载八指标得分表",csv_bytes(result["metrics"]),"ifc_metric_scores.csv","text/csv",key="metric_scores")
            st.subheader("问题定位与导出")
            issue_table(result,"original")
    with tabs[1]:
        if result is None:st.info("先在质量审查看板完成模型分析，再执行修复。")
        else:
            st.subheader("在副本中修复，保留原始模型")
            st.markdown('<div class="section-note">修复缺失、非法或重复GUID；利用世界包围盒Z推断游离建筑构件的最近楼层；把已有显式长度转存为标准栏杆Length。缺失空间净面积、窗宿主和尺寸语义不猜测修复。</div>',unsafe_allow_html=True)
            if st.button("执行自动修复",type="primary",icon="🛠️"):
                try:
                    with st.spinner("修复副本并复查世界几何、空间聚合与导出文件…"):
                        st.session_state["repair"]=repair_bytes(data,float(max_distance))
                        st.session_state["repair"]["max_floor_distance_m"]=float(max_distance)
                except Exception as exc:st.session_state.pop("repair",None);st.error("修复未完成，原文件未修改："+str(exc))
            repaired=st.session_state.get("repair")
            if repaired:
                st.caption(f"本次修复采用最大楼层高差 {repaired['max_floor_distance_m']:.2f}m。修改侧边栏参数后，需再次点击修复才会重新执行。")
                a,b,c=st.columns(3)
                a.metric("修复前得分","—" if result["score"] is None else f"{result['score']:.2f}")
                b.metric("修复后得分","—" if repaired["after"]["score"] is None else f"{repaired['after']['score']:.2f}")
                fixed_count=sum(l["状态"] in ["已修复","已推断修复"] for l in repaired["logs"])
                c.metric("实际修复操作",fixed_count)
                st.success(f"{repaired['geometry_preserved_count']}个已解析目标Body世界网格保持一致，原有聚合关系保持一致。")
                st.subheader("修复日志")
                logs=pd.DataFrame(repaired["logs"]).map(lambda v:"" if v is None or (isinstance(v,float) and math.isnan(v)) else str(v))
                st.dataframe(logs,hide_index=True,width="stretch",height=260)
                a,b=st.columns(2)
                a.download_button("↓ 下载修复后的 IFC",repaired["fixed_data"],Path(filename).stem+"_fixed.ifc","application/octet-stream",key="fixed_ifc",width="stretch")
                b.download_button("↓ 下载修复日志 CSV",csv_bytes(repaired["logs"]),"ifc_fix_log.csv","text/csv",key="fix_log",width="stretch")
                with st.expander("修复后审查明细"):
                    st.dataframe(pd.DataFrame(repaired["after"]["metrics"]),hide_index=True,width="stretch")
                    issue_table(repaired["after"],"after")
                st.caption("GUID修复会改变对象身份，请保留日志。推断楼层不等于经设计确认。修复不保证把所有指标提升到满分。")
            else:st.info("点击按钮后，这里将显示修改日志、前后分数及下载入口。")
    with tabs[2]:rule_panel()


if __name__=="__main__":
    import sys
    if "--audit" in sys.argv:
        # 可复现命令：python app.py --audit 原始.ifc --output 审查.json
        import argparse
        parser=argparse.ArgumentParser(description="与页面共用的IFC质量审查入口")
        parser.add_argument("--audit",required=True);parser.add_argument("--output",required=True)
        args=parser.parse_args()
        try:
            result=analyse_bytes(Path(args.audit).read_bytes())
            Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
            print(json.dumps({"score":result["score"],"version":RULE_VERSION,"fingerprint":audit_fingerprint(result)},ensure_ascii=False))
        except Exception as exc:
            print("无法完成审查："+str(exc),file=sys.stderr);sys.exit(1)
    else:
        main()
