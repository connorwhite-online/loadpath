`sensor-board.kicad_pcb` is a test board for tests/test_kicad.py, built from
footprints of KiCad's official library (gitlab.com/kicad/libraries/kicad-footprints,
CC-BY-SA 4.0 with the library's exception for designs that use it):
MountingHole_2.7mm_M2.5_Pad, USB_C_Receptacle_GCT_USB4085,
QFN-32-1EP_5x5mm_P0.5mm_EP3.45x3.45mm, SOT-223-3_TabPin2, C_0805_2012Metric,
CP_Elec_6.3x7.7, JST_SH_SM04B-SRSS-TB_1x04-1MP_P1.00mm_Horizontal and
PinHeader_1x04_P2.54mm_Vertical. `models/CP_Elec_6.3x7.7.wrl` is a stand-in
box (6.6 mm square, 7.7 mm tall, in KiCad's 0.1 inch VRML units), not
KiCad's model.
