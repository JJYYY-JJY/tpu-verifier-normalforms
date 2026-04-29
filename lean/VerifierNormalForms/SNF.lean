import Lean.Data.Json

namespace VerifierNormalForms

abbrev IntegerMatrix := Array (Array Int)

inductive IntegerMatrixOp where
  | swap (target : Nat) (source : Nat)
  | negate (target : Nat)
  | add (target : Nat) (source : Nat) (scalar : Int)
  deriving BEq, Repr

structure SNFCertificate where
  kind : String
  schemaVersion : String
  rows : Nat
  cols : Nat
  input : IntegerMatrix
  diagonal : IntegerMatrix
  leftTransform : IntegerMatrix
  rightTransform : IntegerMatrix
  rowOps : Array IntegerMatrixOp
  colOps : Array IntegerMatrixOp
  deriving Repr

open Lean

private def snfCertificateKind : String := "snf_int"

private def snfCertificateSchemaVersion : String := "snf-certificate-json-v0.1"

private def requireKindAndSchema (kind schemaVersion : String) : Except String Unit := do
  if kind != snfCertificateKind then
    throw s!"SNF certificate kind must be {snfCertificateKind}"
  if schemaVersion != snfCertificateSchemaVersion then
    throw s!"SNF certificate schema_version must be {snfCertificateSchemaVersion}"
  return ()

private def requireShape (label : String) (rows cols : Nat) (matrix : IntegerMatrix) :
    Except String Unit :=
  if matrix.size == rows && matrix.all (fun row => row.size == cols) then
    return ()
  else
    throw s!"{label} matrix shape mismatch"

private def requireRowIndex (matrix : IntegerMatrix) (row : Nat) : Except String Unit :=
  if row < matrix.size then
    return ()
  else
    throw s!"row index out of bounds: {row}"

private def requireColIndex (columnCount col : Nat) : Except String Unit :=
  if col < columnCount then
    return ()
  else
    throw s!"column index out of bounds: {col}"

private def swapRows (matrix : IntegerMatrix) (target source : Nat) :
    Except String IntegerMatrix := do
  requireRowIndex matrix target
  requireRowIndex matrix source
  let targetRow := matrix[target]!
  let sourceRow := matrix[source]!
  return (matrix.set! target sourceRow).set! source targetRow

private def negateRow (matrix : IntegerMatrix) (target : Nat) :
    Except String IntegerMatrix := do
  requireRowIndex matrix target
  return matrix.set! target (matrix[target]!.map (fun entry => -entry))

private def addRowMultiple
    (matrix : IntegerMatrix) (target source : Nat) (scalar : Int) :
    Except String IntegerMatrix := do
  requireRowIndex matrix target
  requireRowIndex matrix source
  if target == source then
    throw "add row operation requires distinct target and source rows"
  let targetRow := matrix[target]!
  let sourceRow := matrix[source]!
  let row := Array.zipWith (fun targetEntry sourceEntry =>
    targetEntry + scalar * sourceEntry) targetRow sourceRow
  return matrix.set! target row

private def applyRowOp (matrix : IntegerMatrix) : IntegerMatrixOp → Except String IntegerMatrix
  | .swap target source => swapRows matrix target source
  | .negate target => negateRow matrix target
  | .add target source scalar => addRowMultiple matrix target source scalar

private def replayRowOps (matrix : IntegerMatrix) (ops : Array IntegerMatrixOp) :
    Except String IntegerMatrix :=
  ops.foldlM (fun current op => applyRowOp current op) matrix

private def swapCols
    (columnCount : Nat) (matrix : IntegerMatrix) (target source : Nat) :
    Except String IntegerMatrix := do
  requireColIndex columnCount target
  requireColIndex columnCount source
  return matrix.map (fun row =>
    let targetEntry := row[target]!
    let sourceEntry := row[source]!
    (row.set! target sourceEntry).set! source targetEntry)

private def negateCol
    (columnCount : Nat) (matrix : IntegerMatrix) (target : Nat) :
    Except String IntegerMatrix := do
  requireColIndex columnCount target
  return matrix.map (fun row => row.set! target (-row[target]!))

private def addColMultiple
    (columnCount : Nat) (matrix : IntegerMatrix) (target source : Nat) (scalar : Int) :
    Except String IntegerMatrix := do
  requireColIndex columnCount target
  requireColIndex columnCount source
  if target == source then
    throw "add column operation requires distinct target and source columns"
  return matrix.map (fun row =>
    let targetEntry := row[target]!
    let sourceEntry := row[source]!
    row.set! target (targetEntry + scalar * sourceEntry))

private def applyColOp (columnCount : Nat) (matrix : IntegerMatrix) :
    IntegerMatrixOp → Except String IntegerMatrix
  | .swap target source => swapCols columnCount matrix target source
  | .negate target => negateCol columnCount matrix target
  | .add target source scalar => addColMultiple columnCount matrix target source scalar

private def replayColOps
    (columnCount : Nat) (matrix : IntegerMatrix) (ops : Array IntegerMatrixOp) :
    Except String IntegerMatrix :=
  ops.foldlM (fun current op => applyColOp columnCount current op) matrix

private def identityMatrix (size : Nat) : IntegerMatrix :=
  ((List.range size).map (fun row =>
    ((List.range size).map (fun col => if row == col then (1 : Int) else 0)).toArray
  )).toArray

private def requireSNFDiagonal (rows cols : Nat) (matrix : IntegerMatrix) :
    Except String Unit := do
  for rowIndex in List.range rows do
    for colIndex in List.range cols do
      if rowIndex != colIndex && matrix[rowIndex]![colIndex]! != 0 then
        throw "diagonal off-diagonal entries must be zero"
  let diagonalLength := Nat.min rows cols
  let mut previousNonzero? : Option Int := none
  let mut zeroSeen := false
  for index in List.range diagonalLength do
    let entry := matrix[index]![index]!
    if entry < 0 then
      throw "diagonal entries must be nonnegative"
    if entry == 0 then
      zeroSeen := true
    else
      if zeroSeen then
        throw "diagonal entries must remain zero after zero appears"
      match previousNonzero? with
      | some previous =>
          if entry % previous != 0 then
            throw "each nonzero diagonal entry must divide the next one"
      | none => pure ()
      previousNonzero? := some entry

private def multiplyIntegerMatrices
    (left : IntegerMatrix) (leftRows leftCols : Nat)
    (right : IntegerMatrix) (rightRows rightCols : Nat) :
    Except String IntegerMatrix := do
  if leftCols != rightRows then
    throw "matrix multiplication dimension mismatch"
  return ((List.range leftRows).map (fun rowIndex =>
    ((List.range rightCols).map (fun colIndex =>
      (List.range leftCols).foldl
        (fun acc index => acc + left[rowIndex]![index]! * right[index]![colIndex]!)
        (0 : Int)
    )).toArray
  )).toArray

private def joinFailures : List String → String
  | [] => ""
  | first :: rest => rest.foldl (fun acc item => acc ++ "; " ++ item) first

private def getRequiredField (json : Json) (field : String) : Except String Json :=
  match json.getObjVal? field with
  | .ok value => return value
  | .error _ => throw s!"missing SNF certificate field: {field}"

private def getRequiredOpField (json : Json) (field : String) : Except String Json :=
  match json.getObjVal? field with
  | .ok value => return value
  | .error _ => throw s!"operation missing field: {field}"

private def parseShape (json : Json) : Except String (Nat × Nat) := do
  let shape : Array Nat ← fromJson? json
  if shape.size != 2 then
    throw "shape must be [rows, cols]"
  return (shape[0]!, shape[1]!)

private def parseIntegerMatrix (json : Json) : Except String IntegerMatrix :=
  fromJson? json

private def parseIntegerMatrixOp (json : Json) : Except String IntegerMatrixOp := do
  let kind ← (← getRequiredOpField json "kind").getStr?
  match kind with
  | "swap" =>
      let target ← (← getRequiredOpField json "target").getNat?
      let source ← (← getRequiredOpField json "source").getNat?
      return .swap target source
  | "negate" =>
      let target ← (← getRequiredOpField json "target").getNat?
      return .negate target
  | "add" =>
      let target ← (← getRequiredOpField json "target").getNat?
      let source ← (← getRequiredOpField json "source").getNat?
      let scalar ← (← getRequiredOpField json "scalar").getInt?
      return .add target source scalar
  | other => throw s!"unknown integer matrix operation kind: {other}"

private def parseSNFCertificate (json : Json) : Except String SNFCertificate := do
  let kind ← (← getRequiredField json "kind").getStr?
  let schemaVersion ← (← getRequiredField json "schema_version").getStr?
  requireKindAndSchema kind schemaVersion
  let (rows, cols) ← parseShape (← getRequiredField json "shape")
  let input ← parseIntegerMatrix (← getRequiredField json "input")
  let diagonal ← parseIntegerMatrix (← getRequiredField json "diagonal")
  let leftTransform ← parseIntegerMatrix (← getRequiredField json "left_transform")
  let rightTransform ← parseIntegerMatrix (← getRequiredField json "right_transform")
  let rowOpsJson ← (← getRequiredField json "row_ops").getArr?
  let colOpsJson ← (← getRequiredField json "col_ops").getArr?
  let rowOps ← rowOpsJson.mapM parseIntegerMatrixOp
  let colOps ← colOpsJson.mapM parseIntegerMatrixOp
  return {
    kind := kind,
    schemaVersion := schemaVersion,
    rows := rows,
    cols := cols,
    input := input,
    diagonal := diagonal,
    leftTransform := leftTransform,
    rightTransform := rightTransform,
    rowOps := rowOps,
    colOps := colOps
  }

def parseSNFCertificateJson (text : String) : Except String SNFCertificate := do
  let json ← Json.parse text
  parseSNFCertificate json

def verifySNFCertificate (cert : SNFCertificate) : Except String Unit := do
  requireKindAndSchema cert.kind cert.schemaVersion
  requireShape "input" cert.rows cert.cols cert.input
  requireShape "diagonal" cert.rows cert.cols cert.diagonal
  requireShape "left_transform" cert.rows cert.rows cert.leftTransform
  requireShape "right_transform" cert.cols cert.cols cert.rightTransform
  requireSNFDiagonal cert.rows cert.cols cert.diagonal

  let replayedRows ← replayRowOps cert.input cert.rowOps
  let replayed ← replayColOps cert.cols replayedRows cert.colOps
  let expectedLeft ← replayRowOps (identityMatrix cert.rows) cert.rowOps
  let expectedRight ← replayColOps cert.cols (identityMatrix cert.cols) cert.colOps
  let leftInput ← multiplyIntegerMatrices
    cert.leftTransform cert.rows cert.rows cert.input cert.rows cert.cols
  let equationLeft ← multiplyIntegerMatrices
    leftInput cert.rows cert.cols cert.rightTransform cert.cols cert.cols

  let mut failures : Array String := #[]
  if replayed != cert.diagonal then
    failures := failures.push "replay final mismatch"
  if expectedLeft != cert.leftTransform then
    failures := failures.push "left transform mismatch"
  if expectedRight != cert.rightTransform then
    failures := failures.push "right transform mismatch"
  if equationLeft != cert.diagonal then
    failures := failures.push "matrix equation mismatch"

  if failures.isEmpty then
    return ()
  else
    throw ("SNF certificate " ++ joinFailures failures.toList)

def verifySNFCertificateJson (text : String) : Except String Unit := do
  verifySNFCertificate (← parseSNFCertificateJson text)

def checkSNFCertificate (cert : SNFCertificate) : Bool :=
  match verifySNFCertificate cert with
  | .ok _ => true
  | .error _ => false

def checkSNFCertificateJson (text : String) : Bool :=
  match verifySNFCertificateJson text with
  | .ok _ => true
  | .error _ => false

end VerifierNormalForms
