################################################################################
# Copyright 2020-2021 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from ..Component import Component, MAC
from ..DataType import DataType
from ..AsmUtils import vgpr, sgpr

class MAC_F32_Plain(MAC):
    """
    Plain MAC instruction implementation
    """
    @staticmethod
    def asmCaps(caps):
        return caps["v_mac_f32"] or caps["v_fma_f32"]
    #archCaps = {}
    kernel = {"ProblemType": {"DataType": DataType(DataType.single)}}

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.kernel

        instruction = "v_mac_f32"
        if kernel["MACInstruction"] == "FMA":
            if writer.asmCaps["v_fmac_f32"]:
                instruction = "v_fmac_f32"
            elif writer.asmCaps["v_fma_f32"]:
                instruction = "v_fma_f32"
            else:
                raise RuntimeError("FMA instruction specified but not supported on {}".format(kernel["ISA"]))

        if not writer.asmCaps[instruction]:
            raise RuntimeError("{} instruction specified but not supported on {}".format(instruction, kernel["ISA"]))

        kStr = self.commentHeader()

        vars = {}

        vars["m"] = m
        vars["kernel"] = kernel
        vars["endLine"] = writer.endLine

        vars["ThreadTile0"] = kernel["ThreadTile0"]
        vars["ThreadTile1"] = kernel["ThreadTile1"]
        vars["PerformanceWaitCount"] = kernel["PerformanceWaitCount"]

        vars["instruction"] = instruction

        priority = Component.Priority.find(writer)
        macIdx = 0

        if kernel["EnableF32XdlMathOp"]:
            vars["f32MaskStr"] = writer.macXdlF32Mask
            vars["f32IBitsStr"] = str(writer.macXdlF32IBits)
            vars["nanStr"] = vgpr("MacXdlF32Nan")
            vars["incStr"] = vgpr("MacXdlF32Inc")
            vars["tmpStr"] = vgpr("MacXdlF32Tmp")
            vars["maskStr"] = sgpr("MacXdlF32MaskTmp", 2)

        for idx1 in range(0, kernel["ThreadTile1"]):
            for idx0 in range(0, kernel["ThreadTile0"]):
                vars["idx0"] = idx0
                vars["idx1"] = idx1
                vars["a"] = idx0 if writer.tPB["tile01Idx"] else idx1
                vars["b"] = idx1 if writer.tPB["tile01Idx"] else idx0

                if kernel["EnableF32XdlMathOp"]:
                    for iui in range(0, innerUnroll):
                        vars["iui"] = iui

                        vars["aStr"] = "v[vgprValuA_X{m}_I{iui} + {a}]".format_map(vars)
                        vars["bStr"] = "v[vgprValuB_X{m}_I{iui} + {b}]".format_map(vars)

                        kStr += "v_cmp_u_f32 {maskStr}, {aStr}, {aStr}{endLine}".format_map(vars)
                        kStr += "v_bfe_u32 {tmpStr}, {aStr}, {f32IBitsStr}, 1{endLine}".format_map(vars)
                        kStr += "v_add3_u32 {tmpStr}, {aStr}, {tmpStr}, {incStr}{endLine}".format_map(vars)
                        kStr += "v_cndmask_b32 {aStr}, {tmpStr}, {nanStr}, {maskStr}{endLine}".format_map(vars)
                        kStr += "v_and_b32 {aStr}, {f32MaskStr}, {aStr}{endLine}".format_map(vars)

                        kStr += "v_cmp_u_f32 {maskStr}, {bStr}, {bStr}{endLine}".format_map(vars)
                        kStr += "v_bfe_u32 {tmpStr}, {bStr}, {f32IBitsStr}, 1{endLine}".format_map(vars)
                        kStr += "v_add3_u32 {tmpStr}, {bStr}, {tmpStr}, {incStr}{endLine}".format_map(vars)
                        kStr += "v_cndmask_b32 {bStr}, {tmpStr}, {nanStr}, {maskStr}{endLine}".format_map(vars)
                        kStr += "v_and_b32 {bStr}, {f32MaskStr}, {bStr}{endLine}".format_map(vars)


                for iui in range(0, innerUnroll):
                    vars["iui"] = iui

                    vars["cStr"] = "v[vgprValuC + {idx0} + {idx1}*{ThreadTile0}]".format_map(vars)
                    vars["aStr"] = "v[vgprValuA_X{m}_I{iui} + {a}]".format_map(vars)
                    vars["bStr"] = "v[vgprValuB_X{m}_I{iui} + {b}]".format_map(vars)

                    if instruction == "v_fma_f32":
                        kStr += "v_fma_f32 {cStr}, {aStr}, {bStr}, {cStr}{endLine}".format_map(vars)
                    else:
                        kStr += "{instruction} {cStr}, {aStr}, {bStr}{endLine}".format_map(vars)

                    kStr += priority(writer, 1, "Raise priority while processing macs")

                    if macIdx == kernel["PerformanceWaitLocation"]:
                        kStr += "s_waitcnt lgkmcnt({PerformanceWaitCount}) // extra wait for performance{endLine}".format_map(vars)
                    if macIdx == kernel["PerformanceSyncLocation"]:
                        kStr += "s_barrier // extra barrier for performance{endLine}".format_map(vars)
                    macIdx += 1

        kStr += priority(writer, 0, "Reset priority after macs")

        return kStr
