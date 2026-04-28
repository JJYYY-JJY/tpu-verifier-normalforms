import Lean.Data.Json

namespace VerifierNormalForms

abbrev Matrix := Array (Array Int)

structure Pivot where
  row : Nat
  col : Nat
  deriving BEq, Repr

inductive RowOp where
  | swap (target : Nat) (source : Nat)
  | scale (target : Nat) (scalar : Int)
  | add (target : Nat) (source : Nat) (scalar : Int)
  deriving BEq, Repr

structure RREFCertificate where
  modulus : Nat
  rows : Nat
  cols : Nat
  input : Matrix
  ops : Array RowOp
  final : Matrix
  pivots : Array Pivot
  deriving Repr

open Lean

def isPrimeNat (p : Nat) : Bool :=
  p >= 2 && !(List.range (p + 1)).any (fun d => d >= 2 && d * d <= p && p % d == 0)

private def requirePrime (p : Nat) : Except String Unit :=
  if isPrimeNat p then
    return ()
  else
    throw s!"modulus must be prime, got {p}"

private def modp (p : Nat) (x : Int) : Int :=
  x % (Int.ofNat p)

private def normalizeMatrix (p : Nat) (matrix : Matrix) : Matrix :=
  matrix.map (fun row => row.map (modp p))

private def requireShape (label : String) (rows cols : Nat) (matrix : Matrix) :
    Except String Unit :=
  if matrix.size == rows && matrix.all (fun row => row.size == cols) then
    return ()
  else
    throw s!"{label} matrix shape mismatch"

private def requireRowIndex (matrix : Matrix) (row : Nat) : Except String Unit :=
  if row < matrix.size then
    return ()
  else
    throw s!"row index out of bounds: {row}"

private def swapRows (matrix : Matrix) (target source : Nat) : Except String Matrix := do
  requireRowIndex matrix target
  requireRowIndex matrix source
  let targetRow := matrix[target]!
  let sourceRow := matrix[source]!
  return (matrix.set! target sourceRow).set! source targetRow

private def scaleRow (p : Nat) (matrix : Matrix) (target : Nat) (scalar : Int) :
    Except String Matrix := do
  requireRowIndex matrix target
  let factor := modp p scalar
  if factor == 0 then
    throw "row scaling factor must be nonzero modulo p"
  let row := matrix[target]!
  return matrix.set! target (row.map (fun entry => modp p (factor * entry)))

private def addRowMultiple
    (p : Nat) (matrix : Matrix) (target source : Nat) (scalar : Int) :
    Except String Matrix := do
  requireRowIndex matrix target
  requireRowIndex matrix source
  if target == source then
    throw "add row operation requires distinct target and source rows"
  let factor := modp p scalar
  let targetRow := matrix[target]!
  let sourceRow := matrix[source]!
  let row := Array.zipWith (fun targetEntry sourceEntry =>
    modp p (targetEntry + factor * sourceEntry)) targetRow sourceRow
  return matrix.set! target row

private def applyRowOp (p : Nat) (matrix : Matrix) : RowOp → Except String Matrix
  | .swap target source => swapRows matrix target source
  | .scale target scalar => scaleRow p matrix target scalar
  | .add target source scalar => addRowMultiple p matrix target source scalar

private def replayRowOps (p : Nat) (matrix : Matrix) (ops : Array RowOp) :
    Except String Matrix :=
  ops.foldlM (fun current op => applyRowOp p current op) matrix

private def leadingCol (row : Array Int) : Option Nat :=
  (List.range row.size).find? (fun col => row[col]! != 0)

private def deriveRREFPivots (matrix : Matrix) : Except String (Array Pivot) := do
  let mut pivots : Array Pivot := #[]
  let mut previousPivotCol? : Option Nat := none
  let mut seenZeroRow := false
  for rowIndex in List.range matrix.size do
    let row := matrix[rowIndex]!
    match leadingCol row with
    | none =>
        seenZeroRow := true
    | some pivotCol =>
        if seenZeroRow then
          throw "nonzero row appears after zero row"
        match previousPivotCol? with
        | some previousPivotCol =>
            if pivotCol <= previousPivotCol then
              throw "pivot columns must strictly increase"
        | none => pure ()
        if row[pivotCol]! != 1 then
          throw "pivot entry must be one"
        for otherIndex in List.range matrix.size do
          if otherIndex != rowIndex && matrix[otherIndex]![pivotCol]! != 0 then
            throw "pivot column is not reduced"
        pivots := pivots.push { row := rowIndex, col := pivotCol }
        previousPivotCol? := some pivotCol
  return pivots

private def parseShape (json : Json) : Except String (Nat × Nat) := do
  let shape : Array Nat ← fromJson? json
  if shape.size != 2 then
    throw "shape must be [rows, cols]"
  return (shape[0]!, shape[1]!)

private def parseMatrix (json : Json) : Except String Matrix :=
  fromJson? json

private def parsePivot (json : Json) : Except String Pivot := do
  let row ← (← json.getObjVal? "row").getNat?
  let col ← (← json.getObjVal? "col").getNat?
  return { row := row, col := col }

private def parseRowOp (json : Json) : Except String RowOp := do
  let kind ← (← json.getObjVal? "kind").getStr?
  match kind with
  | "swap" =>
      let target ← (← json.getObjVal? "target").getNat?
      let source ← (← json.getObjVal? "source").getNat?
      return .swap target source
  | "scale" =>
      let target ← (← json.getObjVal? "target").getNat?
      let scalar ← (← json.getObjVal? "scalar").getInt?
      return .scale target scalar
  | "add" =>
      let target ← (← json.getObjVal? "target").getNat?
      let source ← (← json.getObjVal? "source").getNat?
      let scalar ← (← json.getObjVal? "scalar").getInt?
      return .add target source scalar
  | other => throw s!"unknown row operation kind: {other}"

private def parseRREFCertificate (json : Json) : Except String RREFCertificate := do
  let kind ← (← json.getObjVal? "kind").getStr?
  if kind != "rref_modp" then
    throw s!"unsupported certificate kind: {kind}"
  let modulus ← (← json.getObjVal? "modulus").getNat?
  let (rows, cols) ← parseShape (← json.getObjVal? "shape")
  let input ← parseMatrix (← json.getObjVal? "input")
  let final ← parseMatrix (← json.getObjVal? "final")
  let opsJson ← (← json.getObjVal? "ops").getArr?
  let ops ← opsJson.mapM parseRowOp
  let pivotsJson ← (← json.getObjVal? "pivots").getArr?
  let pivots ← pivotsJson.mapM parsePivot
  return {
    modulus := modulus,
    rows := rows,
    cols := cols,
    input := input,
    ops := ops,
    final := final,
    pivots := pivots
  }

def parseRREFCertificateJson (text : String) : Except String RREFCertificate := do
  let json ← Json.parse text
  parseRREFCertificate json

def verifyRREFCertificate (cert : RREFCertificate) : Except String Unit := do
  requirePrime cert.modulus
  requireShape "input" cert.rows cert.cols cert.input
  requireShape "final" cert.rows cert.cols cert.final
  let input := normalizeMatrix cert.modulus cert.input
  let final := normalizeMatrix cert.modulus cert.final
  let replayed ← replayRowOps cert.modulus input cert.ops
  if replayed != final then
    throw "replayed matrix does not match final matrix"
  let derivedPivots ← deriveRREFPivots final
  if derivedPivots != cert.pivots then
    throw "supplied pivots do not match final RREF pivots"
  return ()

def verifyRREFCertificateJson (text : String) : Except String Unit := do
  verifyRREFCertificate (← parseRREFCertificateJson text)

def checkRREFCertificate (cert : RREFCertificate) : Bool :=
  match verifyRREFCertificate cert with
  | .ok _ => true
  | .error _ => false

def checkRREFCertificateJson (text : String) : Bool :=
  match verifyRREFCertificateJson text with
  | .ok _ => true
  | .error _ => false

end VerifierNormalForms
