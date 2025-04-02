"""
ImaginaryConsole Emulator - CPU Module
Emulates the ARM Cortex-A78c CPU with 8 cores, ARM64 architecture, ARMv8.2-A version.
"""

import logging
import numpy as np
import threading
from enum import Enum
import time


class CpuMode(Enum):
    """Enum representing CPU operating modes"""
    KERNEL = 0
    USER = 1
    HALTED = 2
    EXCEPTION = 3


class Register:
    """Represents a CPU register"""
    def __init__(self, width=64):
        """Initialize register with given bit width"""
        self.width = width
        self.value = 0
    
    def set(self, value):
        """Set register value with appropriate masking"""
        self.value = value & ((1 << self.width) - 1)
    
    def get(self):
        """Get register value"""
        return self.value


class InstructionType(Enum):
    """Types of ARM64 instructions"""
    ARITHMETIC = 0  # ADD, SUB, etc.
    LOGICAL = 1     # AND, OR, XOR, etc.
    MOVE = 2        # MOV, MVN
    BRANCH = 3      # B, BL, BX, etc.
    MEMORY = 4      # LDR, STR, etc.
    SYSTEM = 5      # SVC, HVC, etc.
    FLOAT = 6       # FADD, FMUL, etc.
    VECTOR = 7      # VADD, VMUL, etc.
    UNKNOWN = 8


class Instruction:
    """Represents a decoded ARM64 instruction"""
    
    def __init__(self, opcode):
        self.opcode = opcode
        self.type = InstructionType.UNKNOWN
        self.mnemonic = "unknown"
        self.operands = []
        
        # Decode the instruction
        self._decode()
    
    def _decode(self):
        """Decode the instruction from its opcode"""
        # Top level decoding based on bits 24-27
        category = (self.opcode >> 24) & 0xF
        
        if category == 0x0:  # 0000 - Data processing immediate
            self._decode_data_processing_imm()
        elif category == 0x1:  # 0001 - Branch, exception, system
            self._decode_branch()
        elif category == 0x2 or category == 0x3:  # 0010/0011 - Loads and stores
            self._decode_load_store()
        elif category >= 0x4 and category <= 0x7:  # 0100-0111 - Data processing register
            self._decode_data_processing_reg()
        elif category == 0x8 or category == 0x9:  # 1000/1001 - Data processing SIMD
            self._decode_data_processing_simd()
        else:  # Other categories
            self.type = InstructionType.UNKNOWN
            self.mnemonic = "unknown"
    
    def _decode_data_processing_imm(self):
        """Decode data processing immediate instructions"""
        op0 = (self.opcode >> 23) & 0x3
        if op0 == 0x0:  # ADD/SUB immediate
            sf = (self.opcode >> 31) & 0x1  # 0 = 32-bit, 1 = 64-bit
            op = (self.opcode >> 30) & 0x1  # 0 = ADD, 1 = SUB
            S = (self.opcode >> 29) & 0x1   # 0 = don't set flags, 1 = set flags
            
            imm12 = (self.opcode >> 10) & 0xFFF
            rn = (self.opcode >> 5) & 0x1F
            rd = self.opcode & 0x1F
            
            self.type = InstructionType.ARITHMETIC
            if op == 0:
                self.mnemonic = "adds" if S else "add"
            else:
                self.mnemonic = "subs" if S else "sub"
            
            self.operands = [rd, rn, imm12]
    
    def _decode_branch(self):
        """Decode branch, exception, and system instructions"""
        op0 = (self.opcode >> 29) & 0x7
        
        if op0 == 0x0 or op0 == 0x4:  # B/BL
            imm26 = self.opcode & 0x3FFFFFF
            # Sign-extend the immediate
            if imm26 & 0x2000000:
                imm26 |= 0xFC000000
            
            self.type = InstructionType.BRANCH
            if op0 == 0x0:
                self.mnemonic = "b"
                self.operands = [imm26 << 2]  # Shift left by 2 to get byte offset
            else:
                self.mnemonic = "bl"
                self.operands = [imm26 << 2]
        
        elif op0 == 0x5:  # Compare & branch
            op = (self.opcode >> 24) & 0x1
            imm19 = (self.opcode >> 5) & 0x7FFFF
            # Sign-extend the immediate
            if imm19 & 0x40000:
                imm19 |= 0xFFF80000
            
            rt = self.opcode & 0x1F
            
            self.type = InstructionType.BRANCH
            if op == 0:
                self.mnemonic = "cbz"
            else:
                self.mnemonic = "cbnz"
            
            self.operands = [rt, imm19 << 2]
    
    def _decode_load_store(self):
        """Decode load/store instructions"""
        size = (self.opcode >> 30) & 0x3
        V = (self.opcode >> 26) & 0x1
        opc = (self.opcode >> 22) & 0x3
        
        rn = (self.opcode >> 5) & 0x1F
        rt = self.opcode & 0x1F
        
        self.type = InstructionType.MEMORY
        
        if (self.opcode >> 27) & 0x1:  # Load/store register (unsigned offset)
            imm12 = (self.opcode >> 10) & 0xFFF
            
            if opc == 0:
                self.mnemonic = "str"
            else:
                self.mnemonic = "ldr"
            
            self.operands = [rt, rn, imm12]
        else:  # Other load/store variations
            self.mnemonic = "ldr" if opc & 0x1 else "str"
            self.operands = [rt, rn]
    
    def _decode_data_processing_reg(self):
        """Decode data processing register instructions"""
        op0 = (self.opcode >> 30) & 0x3
        op1 = (self.opcode >> 28) & 0x3
        op2 = (self.opcode >> 21) & 0xF
        
        rn = (self.opcode >> 5) & 0x1F
        rd = self.opcode & 0x1F
        
        if op0 == 0x0 and op1 == 0x0:  # Logical (shifted register)
            opc = (self.opcode >> 29) & 0x3
            shift = (self.opcode >> 22) & 0x3
            rm = (self.opcode >> 16) & 0x1F
            imm6 = (self.opcode >> 10) & 0x3F
            
            self.type = InstructionType.LOGICAL
            
            if opc == 0x0:
                self.mnemonic = "and"
            elif opc == 0x1:
                self.mnemonic = "bic"
            elif opc == 0x2:
                self.mnemonic = "orr"
            else:
                self.mnemonic = "orn"
            
            self.operands = [rd, rn, rm]
        
        elif op0 == 0x0 and op1 == 0x2:  # Add/subtract (shifted register)
            opc = (self.opcode >> 29) & 0x3
            shift = (self.opcode >> 22) & 0x3
            rm = (self.opcode >> 16) & 0x1F
            imm6 = (self.opcode >> 10) & 0x3F
            
            self.type = InstructionType.ARITHMETIC
            
            if opc & 0x1:  # 0 = ADD, 1 = SUB
                self.mnemonic = "subs" if (opc & 0x2) else "sub"
            else:
                self.mnemonic = "adds" if (opc & 0x2) else "add"
            
            self.operands = [rd, rn, rm]
    
    def _decode_data_processing_simd(self):
        """Decode data processing SIMD instructions"""
        self.type = InstructionType.VECTOR
        self.mnemonic = "simd_op"
        self.operands = []


class CpuCore:
    """Emulates a single ARM CPU core"""
    
    def __init__(self, core_id, architecture="ARM64", version="ARMv8.2-A"):
        """Initialize a CPU core with the given parameters"""
        self.core_id = core_id
        self.architecture = architecture
        self.version = version
        
        # General purpose registers (X0-X30)
        self.registers = [Register(64) for _ in range(31)]
        
        # Special registers
        self.pc = Register(64)  # Program Counter
        self.sp = Register(64)  # Stack Pointer
        self.lr = Register(64)  # Link Register (X30 alias)
        
        # NZCV flags (Negative, Zero, Carry, Overflow)
        self.n_flag = False
        self.z_flag = False
        self.c_flag = False
        self.v_flag = False
        
        # Current CPU mode
        self.mode = CpuMode.HALTED
        
        # Instruction cache
        self.icache = {}
        
        # Instruction execution count
        self.instruction_count = 0
        
        # Current operation
        self.current_operation = "idle"
        
        # Memory reference (to be set by ArmCpu)
        self.memory = None
    
    def reset(self):
        """Reset the CPU core state"""
        for reg in self.registers:
            reg.set(0)
        
        self.pc.set(0)
        self.sp.set(0)
        self.mode = CpuMode.HALTED
        self.instruction_count = 0
        self.current_operation = "idle"
        
        # Clear flags
        self.n_flag = False
        self.z_flag = False
        self.c_flag = False
        self.v_flag = False
    
    def set_pc(self, address):
        """Set the program counter to the given address"""
        self.pc.set(address)
    
    def set_memory(self, memory):
        """Set the memory subsystem reference"""
        self.memory = memory
    
    def fetch_instruction(self):
        """Fetch the instruction at the current PC"""
        if not self.memory:
            return None
            
        # Check if instruction is in cache
        pc = self.pc.get()
        if pc in self.icache:
            return self.icache[pc]
        
        # Fetch from memory (4 bytes for ARM instruction)
        try:
            data = self.memory.read(pc, 4)
            if not data or len(data) < 4:
                return None
                
            # Convert bytes to integer (little-endian)
            opcode = int.from_bytes(data, byteorder='little')
            
            # Decode the instruction
            instruction = Instruction(opcode)
            
            # Cache the instruction
            self.icache[pc] = instruction
            
            return instruction
        except Exception:
            return None
    
    def execute_instruction(self, instruction):
        """Execute a single instruction
        
        Args:
            instruction: Decoded instruction object
            
        Returns:
            True if execution was successful, False otherwise
        """
        if not instruction:
            self.pc.set(self.pc.get() + 4)  # Skip invalid instruction
            return False
        
        executed = False
        
        try:
            # Execute based on instruction type
            if instruction.type == InstructionType.ARITHMETIC:
                executed = self._execute_arithmetic(instruction)
            elif instruction.type == InstructionType.LOGICAL:
                executed = self._execute_logical(instruction)
            elif instruction.type == InstructionType.BRANCH:
                executed = self._execute_branch(instruction)
            elif instruction.type == InstructionType.MEMORY:
                executed = self._execute_memory(instruction)
            elif instruction.type == InstructionType.SYSTEM:
                executed = self._execute_system(instruction)
            elif instruction.type == InstructionType.FLOAT:
                executed = self._execute_float(instruction)
            elif instruction.type == InstructionType.VECTOR:
                executed = self._execute_vector(instruction)
            else:
                # Unknown instruction, just increment PC
                executed = False
            
            # If instruction wasn't handled by specialized execution, increment PC
            if not executed:
                self.pc.set(self.pc.get() + 4)
            
            self.instruction_count += 1
            return True
            
        except Exception:
            # On error, increment PC and continue
            self.pc.set(self.pc.get() + 4)
            return False
    
    def _execute_arithmetic(self, instruction):
        """Execute arithmetic instructions"""
        mnemonic = instruction.mnemonic
        operands = instruction.operands
        
        if mnemonic in ["add", "adds"]:
            # ADD Rd, Rn, imm/Rm
            rd, rn, op2 = operands
            
            # Get source register value
            rn_val = self.registers[rn].get() if rn < len(self.registers) else 0
            
            # Get operand 2 value (immediate or register)
            if isinstance(op2, int):
                op2_val = op2
            else:
                op2_val = self.registers[op2].get() if op2 < len(self.registers) else 0
            
            # Perform addition
            result = rn_val + op2_val
            
            # Set destination register
            if rd < len(self.registers):
                self.registers[rd].set(result)
            
            # Set flags if necessary
            if mnemonic == "adds":
                self.n_flag = (result >> 63) & 1 == 1  # Negative flag - bit 63 is set
                self.z_flag = result == 0              # Zero flag
                self.c_flag = result < rn_val          # Carry flag - result wrapped around
                # V flag is set if both operands have same sign but result has different sign
                self.v_flag = ((rn_val >> 63) == (op2_val >> 63)) and ((rn_val >> 63) != (result >> 63))
            
            # PC is incremented outside
            return False
            
        elif mnemonic in ["sub", "subs"]:
            # SUB Rd, Rn, imm/Rm
            rd, rn, op2 = operands
            
            # Get source register value
            rn_val = self.registers[rn].get() if rn < len(self.registers) else 0
            
            # Get operand 2 value (immediate or register)
            if isinstance(op2, int):
                op2_val = op2
            else:
                op2_val = self.registers[op2].get() if op2 < len(self.registers) else 0
            
            # Perform subtraction
            result = rn_val - op2_val
            
            # Set destination register
            if rd < len(self.registers):
                self.registers[rd].set(result)
            
            # Set flags if necessary
            if mnemonic == "subs":
                self.n_flag = (result >> 63) & 1 == 1  # Negative flag - bit 63 is set
                self.z_flag = result == 0              # Zero flag
                self.c_flag = rn_val >= op2_val        # Carry flag - no borrow occurred
                # V flag is set if operands have different signs and result has different sign from Rn
                self.v_flag = ((rn_val >> 63) != (op2_val >> 63)) and ((rn_val >> 63) != (result >> 63))
            
            # PC is incremented outside
            return False
        
        # Instruction not handled
        return False
    
    def _execute_logical(self, instruction):
        """Execute logical instructions"""
        mnemonic = instruction.mnemonic
        operands = instruction.operands
        
        if mnemonic == "and":
            # AND Rd, Rn, imm/Rm
            rd, rn, op2 = operands
            
            # Get source register value
            rn_val = self.registers[rn].get() if rn < len(self.registers) else 0
            
            # Get operand 2 value (immediate or register)
            if isinstance(op2, int):
                op2_val = op2
            else:
                op2_val = self.registers[op2].get() if op2 < len(self.registers) else 0
            
            # Perform logical AND
            result = rn_val & op2_val
            
            # Set destination register
            if rd < len(self.registers):
                self.registers[rd].set(result)
            
            # PC is incremented outside
            return False
            
        elif mnemonic == "orr":
            # ORR Rd, Rn, imm/Rm
            rd, rn, op2 = operands
            
            # Get source register value
            rn_val = self.registers[rn].get() if rn < len(self.registers) else 0
            
            # Get operand 2 value (immediate or register)
            if isinstance(op2, int):
                op2_val = op2
            else:
                op2_val = self.registers[op2].get() if op2 < len(self.registers) else 0
            
            # Perform logical OR
            result = rn_val | op2_val
            
            # Set destination register
            if rd < len(self.registers):
                self.registers[rd].set(result)
            
            # PC is incremented outside
            return False
        
        # Instruction not handled
        return False
    
    def _execute_branch(self, instruction):
        """Execute branch instructions"""
        mnemonic = instruction.mnemonic
        operands = instruction.operands
        
        if mnemonic == "b":
            # B <label> - branch to offset
            offset = operands[0]
            
            # Calculate target address
            target = self.pc.get() + offset
            
            # Set PC to target
            self.pc.set(target)
            
            # PC is updated, don't increment outside
            return True
            
        elif mnemonic == "bl":
            # BL <label> - branch and link
            offset = operands[0]
            
            # Save return address in link register (X30)
            self.lr.set(self.pc.get() + 4)
            
            # Calculate target address
            target = self.pc.get() + offset
            
            # Set PC to target
            self.pc.set(target)
            
            # PC is updated, don't increment outside
            return True
            
        elif mnemonic in ["cbz", "cbnz"]:
            # CBZ/CBNZ Rt, <label> - compare and branch
            rt, offset = operands
            
            # Get register value
            rt_val = self.registers[rt].get() if rt < len(self.registers) else 0
            
            # Check condition
            if (mnemonic == "cbz" and rt_val == 0) or (mnemonic == "cbnz" and rt_val != 0):
                # Calculate target address
                target = self.pc.get() + offset
                
                # Set PC to target
                self.pc.set(target)
                
                # PC is updated, don't increment outside
                return True
            
            # Condition failed, just increment PC outside
            return False
        
        # Instruction not handled
        return False
    
    def _execute_memory(self, instruction):
        """Execute memory load/store instructions"""
        if not self.memory:
            return False
            
        mnemonic = instruction.mnemonic
        operands = instruction.operands
        
        if mnemonic == "ldr":
            # LDR Rt, [Rn, #imm] - load register
            rt, rn, offset = operands if len(operands) >= 3 else (operands[0], operands[1], 0)
            
            # Get base address
            base_addr = self.registers[rn].get() if rn < len(self.registers) else 0
            
            # Calculate effective address
            addr = base_addr + offset
            
            # Read from memory (8 bytes for 64-bit register)
            data = self.memory.read(addr, 8)
            if not data or len(data) < 8:
                return False
                
            # Convert bytes to integer (little-endian)
            value = int.from_bytes(data, byteorder='little')
            
            # Set destination register
            if rt < len(self.registers):
                self.registers[rt].set(value)
            
            # PC is incremented outside
            return False
            
        elif mnemonic == "str":
            # STR Rt, [Rn, #imm] - store register
            rt, rn, offset = operands if len(operands) >= 3 else (operands[0], operands[1], 0)
            
            # Get base address
            base_addr = self.registers[rn].get() if rn < len(self.registers) else 0
            
            # Calculate effective address
            addr = base_addr + offset
            
            # Get register value
            value = self.registers[rt].get() if rt < len(self.registers) else 0
            
            # Convert integer to bytes (little-endian)
            data = value.to_bytes(8, byteorder='little')
            
            # Write to memory
            self.memory.write(addr, data)
            
            # PC is incremented outside
            return False
        
        # Instruction not handled
        return False
    
    def _execute_system(self, instruction):
        """Execute system instructions"""
        # Not implemented
        return False
    
    def _execute_float(self, instruction):
        """Execute floating-point instructions"""
        # Not implemented
        return False
    
    def _execute_vector(self, instruction):
        """Execute vector/SIMD instructions"""
        # Not implemented
        return False
    
    def get_state(self):
        """Get the current state of the CPU core"""
        return {
            "core_id": self.core_id,
            "pc": self.pc.get(),
            "sp": self.sp.get(),
            "registers": [reg.get() for reg in self.registers],
            "flags": {
                "n": self.n_flag,
                "z": self.z_flag,
                "c": self.c_flag,
                "v": self.v_flag
            },
            "mode": self.mode.name,
            "instruction_count": self.instruction_count
        }
    
    def set_state(self, state):
        """Set the CPU core state from a saved state"""
        self.pc.set(state["pc"])
        self.sp.set(state["sp"])
        
        for i, value in enumerate(state["registers"]):
            if i < len(self.registers):
                self.registers[i].set(value)
        
        self.n_flag = state["flags"]["n"]
        self.z_flag = state["flags"]["z"]
        self.c_flag = state["flags"]["c"]
        self.v_flag = state["flags"]["v"]
        
        self.mode = CpuMode[state["mode"]]
        self.instruction_count = state["instruction_count"]


class ArmCpu:
    """Emulates the ARM Cortex-A78c CPU with multiple cores"""
    
    def __init__(self, cores=8, l3_cache="8MB", architecture="ARM64", version="ARMv8.2-A", frequency=998.4):
        """Initialize the ARM CPU with the given parameters"""
        self.logger = logging.getLogger("ImaginaryConsole.CPU")
        
        self.core_count = cores
        self.l3_cache_size = l3_cache
        self.architecture = architecture
        self.version = version
        
        # Frequency in MHz
        self.frequency = frequency
        
        # Calculate instructions per cycle (IPC) - theoretical value
        self.ipc = 2.5  # Theoretical value for ARM Cortex-A78
        
        # CPU cores
        self.cores = [CpuCore(i, architecture, version) for i in range(cores)]
        
        # Memory reference (to be set by the system)
        self.memory = None
        
        # Thread for background execution
        self.exec_thread = None
        self.running = False
        
        # Performance tracking
        self.performance_counter = 0
        self.performance_tracking = []
        
        self.logger.info(f"Initialized ARM CPU with {cores} cores, {l3_cache} L3 cache, {frequency} MHz")
    
    def set_memory(self, memory):
        """Set the memory subsystem reference"""
        self.memory = memory
        
        # Set memory reference for each core
        for core in self.cores:
            core.set_memory(memory)
    
    def set_frequency(self, frequency):
        """Set the CPU frequency in MHz"""
        self.logger.info(f"Setting CPU frequency to {frequency} MHz")
        self.frequency = frequency
    
    def reset(self):
        """Reset the entire CPU state"""
        for core in self.cores:
            core.reset()
        
        self.performance_counter = 0
        self.performance_tracking = []
        self.logger.info("CPU reset completed")
    
    def set_program_counter(self, address):
        """Set the program counter for all cores"""
        for core in self.cores:
            core.set_pc(address)
    
    def execute_cycle(self, delta_time):
        """Execute a CPU cycle for the given time delta
        
        Args:
            delta_time: Time in seconds for this execution step
        """
        # Number of cycles to execute based on frequency and time
        cycles = int(self.frequency * 1000000 * delta_time)
        
        # Simulate execution of instructions on each core
        for core_id, core in enumerate(self.cores):
            # Only execute if the core is not halted
            if core.mode != CpuMode.HALTED:
                # Calculate instructions to execute based on IPC
                instructions = int(cycles * self.ipc)
                
                # Execute instructions
                for _ in range(instructions):
                    # 1. Fetch the instruction from memory at PC
                    instruction = core.fetch_instruction()
                    
                    # 2. Execute the instruction
                    core.execute_instruction(instruction)
                
                self.performance_counter += instructions
        
        # Sample performance tracking once per second (approximately)
        self.performance_tracking.append(self.performance_counter)
        if len(self.performance_tracking) > 60:  # Keep last 60 samples
            self.performance_tracking.pop(0)
    
    def get_performance(self):
        """Get the current CPU performance statistics"""
        if not self.performance_tracking:
            return 0
        
        # Calculate average instructions per second
        avg_ips = sum(self.performance_tracking) / len(self.performance_tracking)
        return avg_ips
    
    def get_state(self):
        """Get the current state of the entire CPU"""
        return {
            "frequency": self.frequency,
            "cores": [core.get_state() for core in self.cores],
            "performance_counter": self.performance_counter
        }
    
    def set_state(self, state):
        """Set the CPU state from a saved state"""
        self.frequency = state["frequency"]
        
        for i, core_state in enumerate(state["cores"]):
            if i < len(self.cores):
                self.cores[i].set_state(core_state)
        
        self.performance_counter = state["performance_counter"] 