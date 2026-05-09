.intel_syntax noprefix
.text
.global xi_step
.type xi_step, @function

# struct XiState layout (all uint64_t)
# 0   step
# 8   live_word
# 16  carry_event
# 24  A
# 32  theta_ticks
# 40  kappa
# 48  u
# 56  v
# 64  uv
# 72  floor_den
# 80  window_ready
# 88  r_num
# 96  r_den
# 104 cL_num
# 112 cR_num
# 120 c_den
# 128 lowbit

xi_step:
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r15, rdi

    mov rax, [r15 + 0]
    add rax, 1
    mov [r15 + 0], rax

    # primitive live-word law
    mov rax, [r15 + 8]
    mov rcx, rax
    neg rcx
    and rcx, rax
    mov [r15 + 128], rcx
    add rax, rcx
    setc bl
    movzx rbx, bl
    adc rax, 0
    mov [r15 + 8], rax
    mov [r15 + 16], rbx

    # theta / kappa
    mov rax, [r15 + 32]
    add rax, 1
    mov [r15 + 32], rax
    mov rcx, rax
    shr rcx, 2
    mov [r15 + 40], rcx

    # carry => promote articulation class and emit lifted seed (1, A)
    test rbx, rbx
    jz .no_carry_reset
    mov rax, [r15 + 24]
    add rax, 1
    mov [r15 + 24], rax
    mov qword ptr [r15 + 48], 1
    mov [r15 + 56], rax
    mov qword ptr [r15 + 80], 0
.no_carry_reset:

    # balanced arithmetic limb
    mov r12, [r15 + 48]
    mov r13, [r15 + 56]
    mov r14, r12
    imul r14, r13
    mov [r15 + 64], r14

    mov rax, [r15 + 80]
    test rax, rax
    jnz .have_window

    mov rcx, [r15 + 72]
    cmp r14, rcx
    jae .set_window

    # preserve lifted seed on the carry tick; walk corridor starting next tick
    test rbx, rbx
    jnz .have_window

    mov rcx, r12
    mov rdx, r13
    cmp rcx, rdx
    jbe .ordered
    xchg rcx, rdx
.ordered:
    lea r8, [rcx + rdx]
    mov [r15 + 48], rdx
    mov [r15 + 56], r8
    mov r12, rdx
    mov r13, r8
    mov r14, r12
    imul r14, r13
    mov [r15 + 64], r14
    jmp .have_window

.set_window:
    mov qword ptr [r15 + 80], 1

.have_window:
    mov qword ptr [r15 + 88], 1
    mov rax, [r15 + 64]
    mov [r15 + 96], rax

    # completion germ in pi-units
    mov rax, [r15 + 64]
    lea rcx, [rax + rax]
    mov [r15 + 120], rcx

    mov rdx, [r15 + 32]
    imul rdx, rax
    mov r8, rdx
    sub r8, 2
    mov [r15 + 104], r8
    add rdx, 2
    mov [r15 + 112], rdx

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

.section .note.GNU-stack,"",@progbits
