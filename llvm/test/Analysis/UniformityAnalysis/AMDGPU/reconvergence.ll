; RUN: opt -mtriple amdgcn-unknown-amdhsa -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; Test reconvergence points - wave/subgroup intrinsics that force reconvergence

; CHECK-LABEL: for function 'readfirstlane_reconverge':
define amdgpu_kernel void @readfirstlane_reconverge() {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: DIVERGENT:  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cmp = icmp slt i32 %tid, 10
; CHECK: DIVERGENT:  %cmp = icmp slt i32 %tid, 10
  br i1 %cmp, label %then, label %else
; CHECK: DIVERGENT: br i1 %cmp

then:
  %val1 = add i32 %tid, 1
  br label %merge

else:
  %val2 = add i32 %tid, 2
  br label %merge

merge:
  %phi = phi i32 [ %val1, %then ], [ %val2, %else ]
; CHECK: DIVERGENT:  %phi = phi i32
  ; Reconvergence point - readfirstlane forces wave reconvergence
  %reconverged = call i32 @llvm.amdgcn.readfirstlane(i32 %phi)
; CHECK-NOT: DIVERGENT:  %reconverged = call i32 @llvm.amdgcn.readfirstlane
  ; After reconvergence, operations on uniform values should remain uniform
  %uniform_result = add i32 %reconverged, 100
; CHECK-NOT: DIVERGENT:  %uniform_result = add i32
  ret void
}

; CHECK-LABEL: for function 'ballot_reconverge':
define amdgpu_kernel void @ballot_reconverge() {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: DIVERGENT:  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cmp = icmp slt i32 %tid, 32
; CHECK: DIVERGENT:  %cmp = icmp slt i32 %tid, 32
  ; Ballot forces reconvergence across the wave
  %ballot = call i64 @llvm.amdgcn.ballot.i32(i1 %cmp)
; CHECK-NOT: DIVERGENT:  %ballot = call i64 @llvm.amdgcn.ballot.i32
  ; Result of ballot is uniform
  %ballot_trunc = trunc i64 %ballot to i32
; CHECK-NOT: DIVERGENT:  %ballot_trunc = trunc i64 %ballot to i32
  ret void
}

; CHECK-LABEL: for function 'wave_reduce_reconverge':
define amdgpu_kernel void @wave_reduce_reconverge() {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: DIVERGENT:  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  ; Wave reduce forces reconvergence
  %reduced = call i32 @llvm.amdgcn.wave.reduce.umin.i32(i32 %tid)
; CHECK-NOT: DIVERGENT:  %reduced = call i32 @llvm.amdgcn.wave.reduce.umin.i32
  ; Result of reduce is uniform across wave
  %result = add i32 %reduced, 5
; CHECK-NOT: DIVERGENT:  %result = add i32
  ret void
}

; CHECK-LABEL: for function 'icmp_reconverge':
define amdgpu_kernel void @icmp_reconverge() {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK: DIVERGENT:  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  ; Wave comparison forces reconvergence
  %cmp_result = call i64 @llvm.amdgcn.icmp.i32(i32 %tid, i32 10, i32 33)
; CHECK-NOT: DIVERGENT:  %cmp_result = call i64 @llvm.amdgcn.icmp.i32
  ; Result is uniform
  %result = add i64 %cmp_result, 1
; CHECK-NOT: DIVERGENT:  %result = add i64
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i32 @llvm.amdgcn.readfirstlane(i32) #0
declare i64 @llvm.amdgcn.ballot.i32(i1) #1
declare i32 @llvm.amdgcn.wave.reduce.umin.i32(i32) #1
declare i64 @llvm.amdgcn.icmp.i32(i32, i32, i32) #1

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readnone convergent }
