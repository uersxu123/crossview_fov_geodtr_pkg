# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
ET.register_namespace("w", W_NS)


def qn(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def paragraph_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.findall(f".//{qn('t')}"))


def clear_paragraph_text(p: ET.Element, text: str) -> None:
    for child in list(p):
        if child.tag == qn("r"):
            p.remove(child)
    r = ET.SubElement(p, qn("r"))
    t = ET.SubElement(r, qn("t"))
    t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = text


def make_paragraph(template: ET.Element, text: str) -> ET.Element:
    p = copy.deepcopy(template)
    clear_paragraph_text(p, text)
    return p


def main() -> None:
    desktop = Path.home() / "Desktop"
    target = desktop / "三大功能docx.docx"
    if not target.exists():
        matches = [p for p in desktop.glob("*.docx") if p.name.endswith("docx.docx")]
        if len(matches) != 1:
            raise FileNotFoundError(f"无法唯一定位目标 docx：{matches}")
        target = matches[0]

    backup = target.with_name(f"{target.stem}_backup_before_module2{target.suffix}")
    if not backup.exists():
        shutil.copy2(target, backup)

    replacement = [
        "4.1 功能目标",
        "本模块围绕“外业照片拍摄位置识别与地图视角范围映射”开展建设。外业人员上传一张现场照片后，系统结合照片初始位置信息、拍摄方向信息以及对应区域的遥感底图，自动推断照片在地图中的拍摄点、拍摄朝向、视场角和可见范围，并将照片对应的视角范围以扇形或多边形方式叠加到地图/遥感影像上。",
        "当前已完成的工作重点是：实现外业照片到地图的跨视角匹配，自动生成照片在遥感底图上的视场范围，用于判断照片大致覆盖了地图中的哪一片区域，为后续外业核查、图斑定位和人工复核提供空间参考。",
        "4.2 已实现技术流程",
        "第一步：输入外业照片与地图底图",
        "系统读取外业拍摄照片和对应区域的遥感底图/地图截图，同时接收初始拍摄位置、初始方向角、地图比例尺等参数。初始位置可以来自手机 GPS、人工点选或业务系统提供的图上坐标。",
        "第二步：地图候选区域检索",
        "系统以初始位置为中心，在遥感底图上裁剪多个候选地图 patch，并利用跨视角图像检索模型对外业照片与候选地图区域进行相似度匹配，筛选出最可能对应照片拍摄位置的候选区域。",
        "第三步：拍摄姿态优化",
        "在候选区域基础上，系统进一步搜索和优化拍摄点位置、拍摄方向、俯仰角、水平视场角以及地图像素尺度等参数，使照片中的前方视野与地图中的道路走向、边界结构和空间纹理尽可能匹配。",
        "第四步：视角范围生成",
        "系统根据优化后的拍摄点、方向角、视场角和尺度参数，将照片视野投影到遥感底图坐标系中，生成对应的视场范围 polygon，并在地图上绘制拍摄点、中心方向线和视角覆盖区域。",
        "第五步：结果输出",
        "系统输出候选匹配热力图、最佳视角叠加图、候选姿态参数表等结果文件。业务人员可以直观看到外业照片在地图上的拍摄方向和覆盖范围，并据此判断照片与图斑、地块或核查区域之间的空间关系。",
        "4.3 当前阶段输出成果",
        "当前模块已形成一套可运行的原型流程，能够完成“外业照片 → 地图候选检索 → 拍摄姿态估计 → 地图视角范围绘制”的闭环处理。",
        "主要输出包括：",
        "（1）照片在地图中的候选位置热力图；",
        "（2）最佳拍摄视场范围叠加图；",
        "（3）拍摄点坐标、方向角、俯仰角、水平视场角、地图尺度等候选参数表；",
        "（4）用于业务复核的照片视角范围 polygon。",
        "该阶段暂不展开道路、绿化带、田地等具体地物语义分割，重点先解决外业照片与地图之间的空间对应关系，即“这张照片大致是在地图哪里拍的、朝哪个方向拍的、能覆盖地图上的哪一片范围”。",
    ]

    with ZipFile(target, "r") as zin:
        entries = {info.filename: zin.read(info.filename) for info in zin.infolist()}
        infos = zin.infolist()

    root = ET.fromstring(entries["word/document.xml"])
    body = root.find(qn("body"))
    if body is None:
        raise RuntimeError("word/document.xml 中未找到 body")

    children = list(body)
    para_positions = [(idx, child) for idx, child in enumerate(children) if child.tag == qn("p")]
    start_idx = None
    end_idx = None
    for idx, p in para_positions:
        text = paragraph_text(p).strip()
        if text == "四、功能模块二：外业照片智能定位与信息推演":
            start_idx = idx
        elif text.startswith("五、功能模块三") and start_idx is not None:
            end_idx = idx
            break

    if start_idx is None or end_idx is None:
        raise RuntimeError("未能定位功能模块二的起止位置")

    template = children[start_idx + 1]
    new_nodes = [make_paragraph(template, text) for text in replacement]
    body[start_idx + 1 : end_idx] = new_nodes

    entries["word/document.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp_path = Path(tmp.name)

    try:
        with ZipFile(tmp_path, "w", ZIP_DEFLATED) as zout:
            for info in infos:
                data = entries[info.filename]
                zout.writestr(info, data)
        try:
            os.replace(tmp_path, target)
            updated = target
        except PermissionError:
            updated = target.with_name(f"{target.stem}_已更新{target.suffix}")
            os.replace(tmp_path, updated)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print(f"updated={updated}")
    print(f"backup={backup}")


if __name__ == "__main__":
    main()
