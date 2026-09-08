#!/usr/bin/env python3
"""Render the tutorial learner as SVG schematics, using only the standard library.

Run from any directory: python competition/draw_blobs.py
This is a hand-placed schematic of blobs.cir, not a general SPICE layout tool.
Device values and terminal labels come from the netlist; explicit wires are
checked against it, and every device must appear exactly once in the full sheet.
"""
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "competition/examples/blobs.cir"
OUTPUT = ROOT / "docs/figures"


def read_devices():
    devices = {}
    for line in SOURCE.read_text().splitlines():
        if not line.strip() or line.startswith("*"):
            continue
        fields = line.split()
        name = fields[0]
        assert name[0] in "MRC", f"Unsupported device: {line}"
        assert name not in devices, f"Duplicate device: {name}"
        devices[name] = fields
    return devices


class Sheet:
    def __init__(self, devices, width, height, title, description):
        self.devices = devices
        self.drawn = set()
        self.pins = {}
        self.connected = set()
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title description">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="description">{escape(description)}</desc>',
            '<style>text{font-family:Arial,sans-serif;fill:#1f2937}'
            '.net{font-family:monospace;fill:#075985;font-size:14px}'
            '.wire{fill:none;stroke:#334155;stroke-width:2;stroke-linejoin:round}'
            '.note{fill:#526175;font-size:15px}</style>',
            f'<rect width="{width}" height="{height}" fill="white"/>',
        ]

    def text(self, x, y, text, size=17, cls=None, anchor="start"):
        style = f'class="{cls}"' if cls else f'font-size="{size}"'
        self.parts.append(f'<text x="{x}" y="{y}" {style} '
                          f'text-anchor="{anchor}">{escape(text)}</text>')

    def path(self, points):
        self.parts.append('<polyline class="wire" points="' +
                          " ".join(f"{x},{y}" for x, y in points) + '"/>')

    def dot(self, x, y):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="#334155"/>')

    def device(self, name, x, y):
        assert name not in self.drawn, f"Device drawn twice: {name}"
        self.drawn.add(name)
        fields = self.devices[name]
        self.parts.append(f'<g data-device="{name}"><title>{escape(" ".join(fields))}</title>')
        if name.startswith("M"):
            _, drain, gate, source, body, model, *params = fields
            assert model in ("pch", "nch", "syn")
            pm = model == "pch"
            assert body == ("vdd" if pm else "0"), f"Unexpected body: {name}"
            top, bottom = ("S", "D") if pm else ("D", "S")
            nets = {"D": drain, "G": gate, "S": source}
            for pin, point in ((top, (x, y-60)), (bottom, (x, y+60)), ("G", (x-65, y))):
                self.pins[name, pin] = (*point, nets[pin], pin == "G")
            # Simplified MOS symbol: insulated gate, channel, and PMOS gate bubble.
            self.path([(x, y-60), (x, y-25), (x-12, y-25)])
            self.path([(x-12, y-25), (x-12, y+25)])
            self.path([(x-12, y+25), (x, y+25), (x, y+60)])
            self.path([(x-22, y-28), (x-22, y+28)])
            self.path([(x-65, y), (x-(32 if pm else 22), y)])
            if pm:
                self.parts.append(f'<circle cx="{x-27}" cy="{y}" r="5" fill="white" stroke="#334155" stroke-width="2"/>')
            self.text(x+7, y-36, top, 10)
            self.text(x+7, y+43, bottom, 10)
            self.text(x+15, y-7, name, 15)
            self.text(x+15, y+12, model, 13)
            self.text(x+15, y+30, " ".join(params), 11)
        else:
            _, a, b, value = fields
            self.pins[name, "1"] = (x, y-60, a, False)
            self.pins[name, "2"] = (x, y+60, b, False)
            if name.startswith("C"):
                self.path([(x, y-60), (x, y-7)])
                self.path([(x-22, y-7), (x+22, y-7)])
                self.path([(x-22, y+7), (x+22, y+7)])
                self.path([(x, y+7), (x, y+60)])
                value = f"{float(value)*1e6:g} µF"
            else:
                self.path([(x, y-60), (x, y-30)])
                self.parts.append(f'<rect x="{x-9}" y="{y-30}" width="18" height="60" class="wire"/>')
                self.path([(x, y+30), (x, y+60)])
                value += " Ω"
            self.text(x+28, y-5, name, 15)
            self.text(x+28, y+17, value, 14)
        self.parts.append('</g>')

    def wire(self, pins, points=None, label=None):
        terminals = [self.pins[pin] for pin in pins]
        assert len({p[2] for p in terminals}) == 1, f"Shorted nets: {pins}"
        if points is None:
            points = [(p[0], p[1]) for p in terminals]
        for x, y, _, _ in terminals:
            assert any((x1 == x2 == x and min(y1, y2) <= y <= max(y1, y2)) or
                       (y1 == y2 == y and min(x1, x2) <= x <= max(x1, x2))
                       for (x1, y1), (x2, y2) in zip(points, points[1:])), pins
        self.connected.update(pins)
        self.path(points)
        if label:
            self.text(*label, terminals[0][2], cls="net")

    def panel(self, x, y, title, subtitle):
        self.parts.append(f'<rect x="{x+8}" y="{y+8}" width="1104" height="814" rx="12" fill="#f8fafc" stroke="#cbd5e1"/>')
        self.text(x+30, y+42, title, 23)
        self.text(x+30, y+69, subtitle, cls="note")

    def finish(self, path, expected):
        assert self.drawn == set(expected), f"Missing/extra devices: {self.drawn ^ set(expected)}"
        for pin, (x, y, net, gate) in self.pins.items():
            if pin not in self.connected:
                self.text(x-5 if gate else x+5, y+5 if gate else y-6,
                          net, cls="net", anchor="end" if gate else "start")
        self.parts.append('</svg>')
        path.write_text("\n".join(self.parts) + "\n")


def synapse(sheet, c, i, ox=0, oy=0):
    tag = f"{c}_{i}"
    signal = sheet.devices[f"Mpos{tag}"][1]
    sheet.panel(ox, oy, f"Class {c} · {signal} weight cell", "9 transistors + 1 capacitor. Matching blue net labels are electrically connected.")
    def device(prefix, x, y):
        sheet.device(prefix+tag, x+ox, y+oy)
    def wire(pins, points=None, label=None):
        sheet.wire([(prefix+tag, pin) for prefix, pin in pins],
                   None if points is None else [(x+ox, y+oy) for x, y in points],
                   None if label is None else (label[0]+ox, label[1]+oy))
    for prefix, x, y in [
        ("Mu3_", 230, 185), ("Mu4_", 500, 185),
        ("Mu1_", 230, 390), ("Mu2_", 500, 390),
        ("Mtail", 365, 545), ("Menable", 365, 715),
        ("Mreset", 740, 185), ("Cw", 740, 390),
        ("Mpos", 830, 600), ("Mneg", 1020, 600),
    ]:
        device(prefix, x, y)
    wire([("Mu3_", "S"), ("Mu4_", "S")], label=(350, 117))
    wire([("Mu3_", "D"), ("Mu1_", "D")], label=(235, 295))
    wire([("Mu3_", "G"), ("Mu3_", "D")],
         [(165, 185), (140, 185), (140, 270), (230, 270), (230, 245)])
    sheet.dot(230+ox, 270+oy)
    wire([("Mu4_", "D"), ("Mu2_", "D")], label=(505, 295))
    wire([("Mu1_", "S"), ("Mu2_", "S"), ("Mtail", "D")],
         [(230, 450), (230, 470), (500, 470), (500, 450), (500, 470), (365, 470), (365, 485)],
         label=(240, 465))
    sheet.dot(365+ox, 470+oy)
    wire([("Mtail", "S"), ("Menable", "D")], label=(370, 635))
    wire([("Mreset", "S"), ("Cw", "1"), ("Mu4_", "D")],
         [(740, 245), (740, 330), (740, 270), (500, 270), (500, 245)])
    sheet.dot(500+ox, 270+oy)
    sheet.dot(740+ox, 270+oy)
    sheet.text(ox+45, oy+550, "Input-gated", cls="note")
    sheet.text(ox+45, oy+573, "update current", cls="note")
    sheet.text(ox+570, oy+440, "Stored weight", cls="note")
    sheet.text(ox+745, oy+738, "Readout: learned − reference current", cls="note")
    sheet.text(ox+745, oy+762, f"n_pos{c} / n_neg{c} feed the class score circuit.", cls="note")


def readout(sheet, c, ox, oy):
    sheet.panel(ox, oy, f"Class {c} · score and teaching error", "10 transistors + 2 resistors. All three weight cells share n_pos / n_neg / n_error for this class.")
    for prefix, x, y in [
        ("Ma", 180, 380), ("Mc", 440, 380),
        ("Me", 700, 185), ("Md", 700, 380),
        ("Mf", 970, 185), ("Mb", 970, 380),
        ("Rscore", 970, 680),
        ("Mep", 180, 570), ("Men", 180, 735),
        ("Moff", 440, 570), ("Mtarget", 440, 735),
        ("Rerror", 700, 650),
    ]:
        sheet.device(prefix+str(c), x+ox, y+oy)
    def wire(pins, points=None, label=None):
        sheet.wire([(prefix+str(c), pin) for prefix, pin in pins],
                   None if points is None else [(x+ox, y+oy) for x, y in points],
                   None if label is None else (label[0]+ox, label[1]+oy))
    wire([("Me", "D"), ("Md", "D")], label=(705, 290))
    wire([("Mf", "D"), ("Mb", "D")], label=(975, 290))
    for prefix, x, y in [("Ma", 180, 380), ("Mc", 440, 380), ("Me", 700, 185)]:
        drain_y = y+60 if prefix == "Me" else y-60
        wire([(prefix, "G"), (prefix, "D")],
             [(x-65, y), (x-100, y), (x-100, drain_y), (x, drain_y)],
             label=(x-95, drain_y-8))
    wire([("Mep", "D"), ("Men", "D"), ("Moff", "D"), ("Mtarget", "D")],
         [(180, 630), (180, 675), (180, 650), (440, 650), (440, 630), (440, 675)],
         label=(230, 644))
    for x in (180, 440):
        sheet.dot(ox+x, oy+650)
    sheet.text(ox+45, oy+160, "Current mirrors convert", cls="note")
    sheet.text(ox+45, oy+183, "the shared synapse currents", cls="note")
    sheet.text(ox+45, oy+206, f"into a score at out{c}.", cls="note")
    sheet.text(ox+760, oy+540, "The teaching target shifts", cls="note")
    sheet.text(ox+760, oy+563, "the error signal during training.", cls="note")


def heading(sheet, title, subtitle):
    sheet.text(30, 38, title, 27)
    sheet.text(30, 69, subtitle, cls="note")
    sheet.text(30, 95, "MOS: D = drain, S = source; gate enters from left. Bubble = PMOS. Bodies: nch/syn → 0, pch → vdd.", cls="note")
    sheet.text(30, 119, "0 = ground. Named bias, input, target, reset and learn ports are driven by the runner; its sources/loads are not shown.", cls="note")


def main():
    devices = read_devices()
    detail = Sheet(devices, 1120, 980, "One actual learning cell", "Transistor-level schematic of the class 0, input 0 weight cell in blobs.cir.")
    heading(detail, "Inside the two-input learner", "One complete weight cell from competition/examples/blobs.cir. All device names and values come from that file.")
    synapse(detail, 0, 0, oy=145)
    detail.finish(OUTPUT / "blobs_weight_cell.svg", [name for name in devices if name.endswith("0_0")])
    full = Sheet(devices, 2260, 3500, "Complete two-input learning circuit", "All 84 devices in blobs.cir: two class readouts and six weight cells. Matching net labels connect across panels.")
    heading(full, "Two-input analog learner · complete schematic", "74 MOSFETs + 6 capacitors + 4 resistors. Read each class column downward; matching blue net labels connect across all panels.")
    for c in (0, 1):
        readout(full, c, 10+c*1120, 150)
        for i in range(3):
            synapse(full, c, i, 10+c*1120, 980+i*830)
    full.finish(OUTPUT / "blobs_circuit.svg", devices)
    print(f"Rendered {len(full.drawn)} devices; all explicit wires match the netlist.")


if __name__ == "__main__":
    main()
