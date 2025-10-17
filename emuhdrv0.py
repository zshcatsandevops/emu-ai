#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tkinter 600x400 — EMU64 Rel-Edition (Single file)
A compact Tkinter harness + tiny MIPS R4300i (N64) subset that *actually runs* a built-in 8MB test ROM.
- 600×400 fixed GUI
- RDRAM: 8 MiB (big-endian)
- Cart ROM: 8 MiB mapped at physical 0x1000_0000 (mirrored in KSEG0/KSEG1 as usual)
- Simple branch delay-slot toggle
- Micro disassembler and register dump
- I/O "print" port at physical 0x0000_0FF0 (SW low byte emits a character)
- Built-in test ROM prints a banner, does a few ops, loops 5× printing '.', then halts with SYSCALL

This is a teaching toy, not Project64. It does *not* implement the full R4300i, TLB/CP0, RSP/RDP, timing,
or real N64 boot/PI/PIF behavior. The goal is to be self-contained, demonstrably alive, and hackable.

© 2025 FlamesCo & Samsoft — MIT-like for this file: do whatever, no warranty.
"""

import struct
import tkinter as tk
from tkinter import ttk, scrolledtext

# ============================================================
# Constants / Config
# ============================================================
WIN_TITLE = "tkinter 600x400 — EMU64 Rel-Edition"
WIN_W, WIN_H = 600, 400

# We'll boot straight from the cart ROM (KSEG1 mirror) so you can see it run without copy.
# Virtual boot PC chosen to be in KSEG1, mapping to physical 0x1000_0000
BOOT_PC = 0xB0000000  # (pc & 0xF0000000) | (target<<2) logic will keep us in KSEG{0,1}

# IO: a simple "print a character" port in low RAM so a ROM can talk to the GUI.
IO_PRINT_ADDR = 0x00000FF0  # sw low byte emits a character

# 32 GPR names
REG_NAMES = [
    "$zero", "$at", "$v0", "$v1", "$a0", "$a1", "$a2", "$a3",
    "$t0", "$t1", "$t2", "$t3", "$t4", "$t5", "$t6", "$t7",
    "$s0", "$s1", "$s2", "$s3", "$s4", "$s5", "$s6", "$s7",
    "$t8", "$t9", "$k0", "$k1", "$gp", "$sp", "$s8", "$ra",
]

def u32(x): return x & 0xFFFFFFFF
def u64(x): return x & 0xFFFFFFFFFFFFFFFF
def sign16(x):
    x &= 0xFFFF
    return x - 0x10000 if (x & 0x8000) else x

# ============================================================
# Memory
# ============================================================
class N64Memory:
    """
    RDRAM 0x0000_0000 .. 0x007F_FFFF (8 MiB)
    Cart ROM @ physical 0x1000_0000 .. +8 MiB
    KSEG0: 0x8000_0000 .. 0x9FFF_FFFF (cached, direct)   -> phys & 0x1FFF_FFFF
    KSEG1: 0xA000_0000 .. 0xBFFF_FFFF (uncached, direct) -> phys & 0x1FFF_FFFF
    Other regions pass-through as physical (toy simplification).
    """
    RDRAM_SIZE = 8 * 1024 * 1024
    CART_BASE = 0x10000000

    def __init__(self):
        self.rdram = bytearray(self.RDRAM_SIZE)
        self.rom = bytearray(8 * 1024 * 1024)  # 8 MiB demo cart

    def virtual_to_physical(self, address: int) -> int:
        a = address & 0xFFFFFFFF
        if (0x80000000 <= a <= 0xBFFFFFFF):
            return a & 0x1FFFFFFF
        return a  # treat others as physical in this toy

    # ----------- Unaligned-safe loads: do simple byte slicing -----------
    def _be_load32(self, buf: bytearray, p: int) -> int:
        b0 = buf[p:p+4]
        if len(b0) < 4:
            return 0
        return struct.unpack(">I", b0)[0]

    def _be_store32(self, buf: bytearray, p: int, v: int) -> None:
        if 0 <= p <= len(buf) - 4:
            buf[p:p+4] = struct.pack(">I", u32(v))

    def read32(self, address: int) -> int:
        p = self.virtual_to_physical(address)
        # RDRAM window
        if 0 <= p < len(self.rdram):
            return self._be_load32(self.rdram, p)
        # Cart ROM
        if self.CART_BASE <= p < self.CART_BASE + len(self.rom):
            off = p - self.CART_BASE
            return self._be_load32(self.rom, off)
        # Otherwise unmapped -> 0
        return 0

    def write32(self, address: int, value: int) -> None:
        p = self.virtual_to_physical(address)
        # IO: character out (low byte)
        if p == IO_PRINT_ADDR:
            # swallow write to IO, not RAM
            return
        if 0 <= p < len(self.rdram):
            self._be_store32(self.rdram, p, value)
        # Ignore writes to ROM or unmapped

    def read8(self, address: int) -> int:
        p = self.virtual_to_physical(address)
        if 0 <= p < len(self.rdram):
            return self.rdram[p]
        if self.CART_BASE <= p < self.CART_BASE + len(self.rom):
            return self.rom[p - self.CART_BASE]
        return 0

    def write8(self, address: int, value: int) -> None:
        p = self.virtual_to_physical(address)
        if p == IO_PRINT_ADDR:
            return
        if 0 <= p < len(self.rdram):
            self.rdram[p] = value & 0xFF

    def store_words(self, vaddr: int, words) -> None:
        """Write a sequence of 32-bit words to RDRAM starting at vaddr (big-endian)."""
        p = self.virtual_to_physical(vaddr)
        for i, w in enumerate(words):
            off = p + i * 4
            self._be_store32(self.rdram, off, w)

# ============================================================
# CPU (tiny subset interpreter)
# ============================================================
class MIPSR4300i:
    def __init__(self, mem: N64Memory):
        self.memory = mem
        self.gpr = [0] * 32
        self.hi = 0
        self.lo = 0
        self.pc = BOOT_PC
        self.next_pc = self.pc + 4
        self.cycles = 0
        self.delay_slots = True  # toggle in GUI
        self.halted = False
        # IO callback: callable(kind, value) -> None
        self.io_write = None

    # --------------- Core control ---------------
    def reset(self):
        self.gpr = [0] * 32
        self.hi = 0
        self.lo = 0
        self.pc = BOOT_PC
        self.next_pc = self.pc + 4
        self.cycles = 0
        self.halted = False
        self.gpr[29] = 0x00100000  # a tiny stack, just for show

    def fetch(self, address=None) -> int:
        return self.memory.read32(self.pc if address is None else address)

    def step(self):
        """One architectural step with optional branch-delay slot execution."""
        if self.halted:
            return
        ins = self.fetch()
        default_next = (self.pc + 4) & 0xFFFFFFFF
        self.next_pc = default_next

        # Execute; may change next_pc
        self.decode_execute(ins)

        branch_taken = (self.next_pc != default_next)
        branch_target = self.next_pc

        # Optional delay slot
        if self.delay_slots and branch_taken:
            slot_pc = default_next
            slot_ins = self.memory.read32(slot_pc)
            saved_pc, saved_next = self.pc, self.next_pc
            self.pc = slot_pc
            self.next_pc = (slot_pc + 4) & 0xFFFFFFFF
            self.decode_execute(slot_ins)
            # Restore branch target
            self.pc = saved_pc
            self.next_pc = branch_target

        # Commit PC
        self.pc = self.next_pc & 0xFFFFFFFF
        self.next_pc = (self.pc + 4) & 0xFFFFFFFF
        self.cycles += 1
        self.gpr[0] = 0  # hard-wired zero

    # --------------- Decode / Execute ---------------
    def decode_execute(self, ins: int):
        op  = (ins >> 26) & 0x3F
        rs  = (ins >> 21) & 0x1F
        rt  = (ins >> 16) & 0x1F
        rd  = (ins >> 11) & 0x1F
        sh  = (ins >> 6)  & 0x1F
        fn  = ins & 0x3F
        imm = ins & 0xFFFF
        tgt = ins & 0x03FFFFFF
        imm_se = sign16(imm)
        self.gpr[0] = 0

        if op == 0x00:  # SPECIAL
            self._special(rs, rt, rd, sh, fn, ins)
        elif op == 0x02:  # J
            self.next_pc = ((self.pc & 0xF0000000) | (tgt << 2)) & 0xFFFFFFFF
        elif op == 0x03:  # JAL
            self.gpr[31] = u64(self.pc + 8)
            self.next_pc = ((self.pc & 0xF0000000) | (tgt << 2)) & 0xFFFFFFFF
        elif op == 0x04:  # BEQ
            if self.gpr[rs] == self.gpr[rt]:
                self.next_pc = (self.pc + 4 + (imm_se << 2)) & 0xFFFFFFFF
        elif op == 0x05:  # BNE
            if self.gpr[rs] != self.gpr[rt]:
                self.next_pc = (self.pc + 4 + (imm_se << 2)) & 0xFFFFFFFF
        elif op == 0x08 or op == 0x09:  # ADDI / ADDIU
            self.gpr[rt] = u64(self.gpr[rs] + imm_se)
        elif op == 0x0C:  # ANDI
            self.gpr[rt] = u64(self.gpr[rs] & (imm & 0xFFFF))
        elif op == 0x0D:  # ORI
            self.gpr[rt] = u64(self.gpr[rs] | (imm & 0xFFFF))
        elif op == 0x0F:  # LUI
            self.gpr[rt] = u64((imm & 0xFFFF) << 16)
        elif op == 0x20:  # LB (toy, sign-extended from mem8)
            self.gpr[rt] = u64((self.memory.read8(self.gpr[rs] + imm_se) ^ 0x80) - 0x80)
        elif op == 0x23:  # LW
            self.gpr[rt] = u64(self.memory.read32(self.gpr[rs] + imm_se))
        elif op == 0x28:  # SB
            self.memory.write8(self.gpr[rs] + imm_se, self.gpr[rt])
            # IO hook for byte, too
            if hasattr(self, "io_write") and self.io_write and ((self.gpr[rs] + imm_se) & 0xFFFFFFFF) == IO_PRINT_ADDR:
                self.io_write("char", self.gpr[rt] & 0xFF)
        elif op == 0x2B:  # SW
            addr = (self.gpr[rs] + imm_se) & 0xFFFFFFFF
            # IO hook: low byte of SW emits a char
            if hasattr(self, "io_write") and self.io_write and addr == IO_PRINT_ADDR:
                self.io_write("char", self.gpr[rt] & 0xFF)
            self.memory.write32(addr, self.gpr[rt])
        # (others omitted in this compact core)
        self.gpr[0] = 0

    def _special(self, rs, rt, rd, sh, fn, ins):
        if fn == 0x00:   # SLL
            self.gpr[rd] = u64(self.gpr[rt] << sh)
        elif fn == 0x02: # SRL
            self.gpr[rd] = u64((self.gpr[rt] & 0xFFFFFFFFFFFFFFFF) >> sh)
        elif fn == 0x08: # JR
            self.next_pc = self.gpr[rs] & 0xFFFFFFFF
        elif fn == 0x09: # JALR
            self.gpr[rd] = u64(self.pc + 8)
            self.next_pc = self.gpr[rs] & 0xFFFFFFFF
        elif fn == 0x0C: # SYSCALL (toy: halt)
            self.halted = True
        elif fn == 0x12: # MFLO
            self.gpr[rd] = u64(self.lo)
        elif fn == 0x18: # MULT (32×32 -> 64)
            a = u32(self.gpr[rs])
            b = u32(self.gpr[rt])
            r = (a * b) & 0xFFFFFFFFFFFFFFFF
            self.lo = r & 0xFFFFFFFF
            self.hi = (r >> 32) & 0xFFFFFFFF
        elif fn == 0x20 or fn == 0x21:  # ADD/ADDU
            self.gpr[rd] = u64(self.gpr[rs] + self.gpr[rt])
        elif fn == 0x22 or fn == 0x23:  # SUB/SUBU
            self.gpr[rd] = u64(self.gpr[rs] - self.gpr[rt])
        elif fn == 0x24: # AND
            self.gpr[rd] = u64(self.gpr[rs] & self.gpr[rt])
        elif fn == 0x25: # OR
            self.gpr[rd] = u64(self.gpr[rs] | self.gpr[rt])
        # (others omitted)

# ============================================================
# Tiny assembler / disassembler (subset for demo + tests)
# ============================================================
def enc_r(rs=0, rt=0, rd=0, sh=0, fn=0, op=0):
    return ((op & 0x3F) << 26) | ((rs & 0x1F) << 21) | ((rt & 0x1F) << 16) | \
           ((rd & 0x1F) << 11) | ((sh & 0x1F) << 6) | (fn & 0x3F)

def enc_i(op, rs, rt, imm):
    return ((op & 0x3F) << 26) | ((rs & 0x1F) << 21) | ((rt & 0x1F) << 16) | (imm & 0xFFFF)

def enc_j(op, target_addr):
    return ((op & 0x3F) << 26) | (((target_addr >> 2) & 0x03FFFFFF))

def enc_syscall(code=0):
    return ((code & 0xFFFFF) << 6) | 0x0000000C  # SPECIAL with fn=0x0C

def disasm(w, pc=0):
    op  = (w >> 26) & 0x3F
    rs  = (w >> 21) & 0x1F
    rt  = (w >> 16) & 0x1F
    rd  = (w >> 11) & 0x1F
    sh  = (w >> 6)  & 0x1F
    fn  = w & 0x3F
    imm = w & 0xFFFF
    tgt = w & 0x03FFFFFF
    if op == 0x00:
        if fn == 0x00:  return f"sll {REG_NAMES[rd]}, {REG_NAMES[rt]}, {sh}"
        if fn == 0x02:  return f"srl {REG_NAMES[rd]}, {REG_NAMES[rt]}, {sh}"
        if fn == 0x08:  return f"jr {REG_NAMES[rs]}"
        if fn == 0x09:  return f"jalr {REG_NAMES[rd]}, {REG_NAMES[rs]}"
        if fn == 0x0C:  return "syscall"
        if fn == 0x12:  return f"mflo {REG_NAMES[rd]}"
        if fn == 0x18:  return f"mult {REG_NAMES[rs]}, {REG_NAMES[rt]}"
        if fn in (0x20, 0x21): return f"addu {REG_NAMES[rd]}, {REG_NAMES[rs]}, {REG_NAMES[rt]}"
        if fn in (0x22, 0x23): return f"subu {REG_NAMES[rd]}, {REG_NAMES[rs]}, {REG_NAMES[rt]}"
        if fn == 0x24:  return f"and {REG_NAMES[rd]}, {REG_NAMES[rs]}, {REG_NAMES[rt]}"
        if fn == 0x25:  return f"or {REG_NAMES[rd]}, {REG_NAMES[rs]}, {REG_NAMES[rt]}"
        return f"special 0x{fn:02X}"
    if op == 0x02:       return f"j 0x{(((pc & 0xF0000000) | (tgt << 2)) & 0xFFFFFFFF):08X}"
    if op == 0x03:       return f"jal 0x{(((pc & 0xF0000000) | (tgt << 2)) & 0xFFFFFFFF):08X}"
    if op == 0x04:       return f"beq {REG_NAMES[rs]}, {REG_NAMES[rt]}, {sign16(imm)}"
    if op == 0x05:       return f"bne {REG_NAMES[rs]}, {REG_NAMES[rt]}, {sign16(imm)}"
    if op == 0x08:       return f"addi {REG_NAMES[rt]}, {REG_NAMES[rs]}, {sign16(imm)}"
    if op == 0x09:       return f"addiu {REG_NAMES[rt]}, {REG_NAMES[rs]}, {sign16(imm)}"
    if op == 0x0C:       return f"andi {REG_NAMES[rt]}, {REG_NAMES[rs]}, 0x{imm:04X}"
    if op == 0x0D:       return f"ori {REG_NAMES[rt]}, {REG_NAMES[rs]}, 0x{imm:04X}"
    if op == 0x0F:       return f"lui {REG_NAMES[rt]}, 0x{imm:04X}"
    if op == 0x20:       return f"lb {REG_NAMES[rt]}, {sign16(imm)}({REG_NAMES[rs]})"
    if op == 0x23:       return f"lw {REG_NAMES[rt]}, {sign16(imm)}({REG_NAMES[rs]})"
    if op == 0x28:       return f"sb {REG_NAMES[rt]}, {sign16(imm)}({REG_NAMES[rs]})"
    if op == 0x2B:       return f"sw {REG_NAMES[rt]}, {sign16(imm)}({REG_NAMES[rs]})"
    return f"op 0x{op:02X}"

# ============================================================
# Built-in 8 MiB test ROM (no file I/O)
# ============================================================
def build_test_rom_8mb() -> bytearray:
    """
    Compose a tiny program that:
      - prints "Hello from ROM!\\n"
      - stores 0x12345678 at RAM[0], loads back, adds, etc.
      - loops 5× printing '.' (shows branch + delay slot behavior)
      - halts with SYSCALL
    Placed at the very beginning of cart ROM (phys 0x1000_0000), executed via BOOT_PC in KSEG1.
    """
    words = []

    def P(w): words.append(w & 0xFFFFFFFF)

    # Print banner
    for ch in b"Hello from ROM!\\n":
        P(enc_i(0x0D, 0, 8, ch))                 # ori  $t0,$zero,ch
        P(enc_i(0x2B, 0, 8, IO_PRINT_ADDR))      # sw   $t0, IO($zero)

    # Memory touch: t0 = 0x12345678; store, load, add
    P(enc_i(0x0F, 0, 8, 0x1234))                 # lui  $t0,0x1234
    P(enc_i(0x0D, 8, 8, 0x5678))                 # ori  $t0,$t0,0x5678
    P(enc_i(0x2B, 0, 8, 0))                      # sw   $t0,0($zero)
    P(enc_i(0x23, 0, 9, 0))                      # lw   $t1,0($zero)
    P(enc_r(8, 9, 10, 0, 0x21))                  # addu $t2,$t0,$t1

    # Loop 5×: t3 = 0; t4 = 5; loop: t3+=1; print '.'; bne t3,t4, loop
    P(enc_i(0x0D, 0, 11, 0))                     # ori  $t3,$zero,0
    P(enc_i(0x0D, 0, 12, 5))                     # ori  $t4,$zero,5
    loop_here = len(words)                        # label
    P(enc_i(0x08, 11, 11, 1))                    # addi $t3,$t3,1
    P(enc_i(0x0D, 0, 8, ord('.')))               # ori  $t0,$zero,'.'
    P(enc_i(0x2B, 0, 8, IO_PRINT_ADDR))          # sw   $t0, IO($zero)
    P(enc_i(0x05, 11, 12, (-4) & 0xFFFF))        # bne  $t3,$t4, loop (back 4 instrs)
    P(0x00000000)                                # nop (delay slot)

    # Done
    P(enc_syscall(0))                            # syscall -> halt

    rom = bytearray(8 * 1024 * 1024)
    # Write program at start of ROM (phys 0x1000_0000)
    off = 0
    for w in words:
        if off + 4 > len(rom): break
        rom[off:off+4] = struct.pack(">I", w)
        off += 4
    return rom

# ============================================================
# GUI
# ============================================================
class EMU64RelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(WIN_TITLE)
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.root.resizable(False, False)

        # Core
        self.mem = N64Memory()
        self.cpu = MIPSR4300i(self.mem)
        self.cpu.io_write = self._io_write  # hook for IO print
        self.running = False
        self.steps_per_tick = 128

        # Text console
        self.text = scrolledtext.ScrolledText(
            root, width=70, height=22, bg="#101010", fg="#00FF88",
            insertbackground="#00FF88", font=("Consolas", 10)
        )
        self.text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Toolbar
        bar = ttk.Frame(root); bar.pack(fill=tk.X, padx=4, pady=(0,4))
        ttk.Button(bar, text="Reset", command=self.reset_cpu).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="Step", command=self.step_once).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="Run/Pause", command=self.toggle_run).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="Disasm @PC", command=self.disasm_here).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="Dump Regs", command=self.dump_regs).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="Load 8MB Test ROM", command=self.load_test_rom).pack(side=tk.RIGHT, padx=3)
        self.delay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Delay slots", variable=self.delay_var, command=self.on_delay_toggle).pack(side=tk.RIGHT)

        # Status
        self.status = ttk.Label(root, text="", anchor="w")
        self.status.pack(fill=tk.X, padx=6, pady=(0,6))

        self.println("EMU64 Rel-Edition initialized.")
        self.println("Built-in 8MB test ROM is available. Click [Load 8MB Test ROM], then Run/Step.")
        self.update_status()

    # ---------- UI helpers ----------
    def println(self, s): 
        self.text.insert(tk.END, s + "\n")
        self.text.see(tk.END)

    def update_status(self):
        self.status.config(text=f"PC=0x{self.cpu.pc:08X}   Cycles={self.cpu.cycles}   "
                                f"Running={'Yes' if self.running else 'No'}   "
                                f"DelaySlots={'On' if self.cpu.delay_slots else 'Off'}   "
                                f"Halted={'Yes' if self.cpu.halted else 'No'}")

    def _io_write(self, kind, value):
        if kind == "char":
            ch = value & 0xFF
            if ch == 10:
                self.println("")
            else:
                # insert without forcing newline
                self.text.insert(tk.END, chr(ch))
                self.text.see(tk.END)

    # ---------- Actions ----------
    def reset_cpu(self):
        self.cpu.reset()
        self.println("CPU Reset.")
        self.update_status()

    def step_once(self):
        if self.cpu.halted:
            self.println("⚠️  CPU is halted. Reset or load ROM.")
            self.update_status()
            return
        pc = self.cpu.pc
        ins = self.cpu.fetch()
        self.println(f"{pc:08X}: {ins:08X}   {disasm(ins, pc=pc)}")
        self.cpu.step()
        if self.cpu.halted:
            self.println("HALT (syscall).")
        self.update_status()

    def toggle_run(self):
        self.running = not self.running
        if self.running:
            self.println("Running…")
            self._tick()
        else:
            self.println("Paused.")
        self.update_status()

    def _tick(self):
        if not self.running:
            return
        if self.cpu.halted:
            self.println("HALT (syscall).")
            self.running = False
            self.update_status()
            return
        for _ in range(self.steps_per_tick):
            self.cpu.step()
            if self.cpu.halted:
                break
        self.update_status()
        # Keep it snappy without freezing the UI
        self.root.after(1, self._tick)

    def on_delay_toggle(self):
        self.cpu.delay_slots = bool(self.delay_var.get())
        self.update_status()

    def disasm_here(self, count=8):
        base = self.cpu.pc
        for i in range(count):
            addr = (base + i * 4) & 0xFFFFFFFF
            w = self.mem.read32(addr)
            self.println(f"{addr:08X}: {w:08X}   {disasm(w, pc=addr)}")
        self.update_status()

    def dump_regs(self):
        cols = []
        for i in range(0, 32, 4):
            part = "  ".join(f"{REG_NAMES[i+j]:>4}={self.cpu.gpr[i+j]:016X}" for j in range(4))
            cols.append(part)
        self.println("\n".join(cols))
        self.update_status()

    def load_test_rom(self):
        """Build and map an 8MB in-memory test ROM, then reset CPU to BOOT_PC (KSEG1)."""
        self.mem.rom = build_test_rom_8mb()
        self.cpu.reset()
        self.println("8MB Test ROM loaded @ phys 0x1000_0000 (virt 0xB000_0000). Disassemble and Run/Step to watch it go.")
        self.update_status()

# ============================================================
def main():
    root = tk.Tk()
    EMU64RelGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
 
