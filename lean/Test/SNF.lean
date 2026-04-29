import VerifierNormalForms

namespace Test.SNF

def validJson : String :=
  "{\"kind\":\"snf_int\",\"schema_version\":\"snf-certificate-json-v0.1\",\"shape\":[2,2],\"input\":[[2,4],[6,8]],\"diagonal\":[[2,0],[0,4]],\"left_transform\":[[1,0],[3,-1]],\"right_transform\":[[1,-2],[0,1]],\"row_ops\":[{\"kind\":\"add\",\"target\":1,\"source\":0,\"scalar\":-3},{\"kind\":\"negate\",\"target\":1}],\"col_ops\":[{\"kind\":\"add\",\"target\":1,\"source\":0,\"scalar\":-2}]}"

def wrongKindJson : String :=
  "{\"kind\":\"rref_modp\",\"schema_version\":\"snf-certificate-json-v0.1\",\"shape\":[1,1],\"input\":[[1]],\"diagonal\":[[1]],\"left_transform\":[[1]],\"right_transform\":[[1]],\"row_ops\":[],\"col_ops\":[]}"

def wrongSchemaJson : String :=
  "{\"kind\":\"snf_int\",\"schema_version\":\"v0\",\"shape\":[1,1],\"input\":[[1]],\"diagonal\":[[1]],\"left_transform\":[[1]],\"right_transform\":[[1]],\"row_ops\":[],\"col_ops\":[]}"

def missingSourceJson : String :=
  "{\"kind\":\"snf_int\",\"schema_version\":\"snf-certificate-json-v0.1\",\"shape\":[1,1],\"input\":[[1]],\"diagonal\":[[1]],\"left_transform\":[[1]],\"right_transform\":[[1]],\"row_ops\":[{\"kind\":\"swap\",\"target\":0}],\"col_ops\":[]}"

def missingScalarJson : String :=
  "{\"kind\":\"snf_int\",\"schema_version\":\"snf-certificate-json-v0.1\",\"shape\":[1,1],\"input\":[[1]],\"diagonal\":[[1]],\"left_transform\":[[1]],\"right_transform\":[[1]],\"row_ops\":[{\"kind\":\"add\",\"target\":0,\"source\":0}],\"col_ops\":[]}"

def unknownOpJson : String :=
  "{\"kind\":\"snf_int\",\"schema_version\":\"snf-certificate-json-v0.1\",\"shape\":[1,1],\"input\":[[1]],\"diagonal\":[[1]],\"left_transform\":[[1]],\"right_transform\":[[1]],\"row_ops\":[{\"kind\":\"scale\",\"target\":0,\"scalar\":1}],\"col_ops\":[]}"

def rowAddSelfJson : String :=
  "{\"kind\":\"snf_int\",\"schema_version\":\"snf-certificate-json-v0.1\",\"shape\":[1,1],\"input\":[[1]],\"diagonal\":[[2]],\"left_transform\":[[2]],\"right_transform\":[[1]],\"row_ops\":[{\"kind\":\"add\",\"target\":0,\"source\":0,\"scalar\":1}],\"col_ops\":[]}"

def colOutOfBoundsJson : String :=
  "{\"kind\":\"snf_int\",\"schema_version\":\"snf-certificate-json-v0.1\",\"shape\":[1,1],\"input\":[[1]],\"diagonal\":[[1]],\"left_transform\":[[1]],\"right_transform\":[[1]],\"row_ops\":[],\"col_ops\":[{\"kind\":\"swap\",\"target\":0,\"source\":1}]}"

def diagonalMismatchJson : String :=
  "{\"kind\":\"snf_int\",\"schema_version\":\"snf-certificate-json-v0.1\",\"shape\":[2,2],\"input\":[[2,4],[6,8]],\"diagonal\":[[2,0],[0,8]],\"left_transform\":[[1,0],[3,-1]],\"right_transform\":[[1,-2],[0,1]],\"row_ops\":[{\"kind\":\"add\",\"target\":1,\"source\":0,\"scalar\":-3},{\"kind\":\"negate\",\"target\":1}],\"col_ops\":[{\"kind\":\"add\",\"target\":1,\"source\":0,\"scalar\":-2}]}"

def leftTransformMismatchJson : String :=
  "{\"kind\":\"snf_int\",\"schema_version\":\"snf-certificate-json-v0.1\",\"shape\":[2,2],\"input\":[[2,4],[6,8]],\"diagonal\":[[2,0],[0,4]],\"left_transform\":[[1,0],[2,-1]],\"right_transform\":[[1,-2],[0,1]],\"row_ops\":[{\"kind\":\"add\",\"target\":1,\"source\":0,\"scalar\":-3},{\"kind\":\"negate\",\"target\":1}],\"col_ops\":[{\"kind\":\"add\",\"target\":1,\"source\":0,\"scalar\":-2}]}"

def rightTransformMismatchJson : String :=
  "{\"kind\":\"snf_int\",\"schema_version\":\"snf-certificate-json-v0.1\",\"shape\":[2,2],\"input\":[[2,4],[6,8]],\"diagonal\":[[2,0],[0,4]],\"left_transform\":[[1,0],[3,-1]],\"right_transform\":[[1,-1],[0,1]],\"row_ops\":[{\"kind\":\"add\",\"target\":1,\"source\":0,\"scalar\":-3},{\"kind\":\"negate\",\"target\":1}],\"col_ops\":[{\"kind\":\"add\",\"target\":1,\"source\":0,\"scalar\":-2}]}"

def equationMismatchJson : String :=
  "{\"kind\":\"snf_int\",\"schema_version\":\"snf-certificate-json-v0.1\",\"shape\":[1,1],\"input\":[[2]],\"diagonal\":[[2]],\"left_transform\":[[1]],\"right_transform\":[[-1]],\"row_ops\":[],\"col_ops\":[]}"

def offDiagonalNonzeroJson : String :=
  "{\"kind\":\"snf_int\",\"schema_version\":\"snf-certificate-json-v0.1\",\"shape\":[2,2],\"input\":[[2,1],[0,4]],\"diagonal\":[[2,1],[0,4]],\"left_transform\":[[1,0],[0,1]],\"right_transform\":[[1,0],[0,1]],\"row_ops\":[],\"col_ops\":[]}"

def divisibilityViolationJson : String :=
  "{\"kind\":\"snf_int\",\"schema_version\":\"snf-certificate-json-v0.1\",\"shape\":[2,2],\"input\":[[4,0],[0,6]],\"diagonal\":[[4,0],[0,6]],\"left_transform\":[[1,0],[0,1]],\"right_transform\":[[1,0],[0,1]],\"row_ops\":[],\"col_ops\":[]}"

def zeroRowJson : String :=
  "{\"kind\":\"snf_int\",\"schema_version\":\"snf-certificate-json-v0.1\",\"shape\":[0,2],\"input\":[],\"diagonal\":[],\"left_transform\":[],\"right_transform\":[[1,-1],[0,1]],\"row_ops\":[],\"col_ops\":[{\"kind\":\"add\",\"target\":1,\"source\":0,\"scalar\":-1}]}"

def zeroColJson : String :=
  "{\"kind\":\"snf_int\",\"schema_version\":\"snf-certificate-json-v0.1\",\"shape\":[2,0],\"input\":[[],[]],\"diagonal\":[[],[]],\"left_transform\":[[1,0],[1,1]],\"right_transform\":[],\"row_ops\":[{\"kind\":\"add\",\"target\":1,\"source\":0,\"scalar\":1}],\"col_ops\":[]}"

def expectOk : Except String Unit → Bool
  | .ok _ => true
  | .error _ => false

def expectError : Except String Unit → Bool
  | .ok _ => false
  | .error _ => true

#guard match VerifierNormalForms.parseSNFCertificateJson validJson with
  | .ok cert => VerifierNormalForms.checkSNFCertificate cert
  | .error _ => false

#guard expectOk (VerifierNormalForms.verifySNFCertificateJson validJson)
#guard VerifierNormalForms.checkSNFCertificateJson validJson
#guard expectError (VerifierNormalForms.verifySNFCertificateJson wrongKindJson)
#guard expectError (VerifierNormalForms.verifySNFCertificateJson wrongSchemaJson)
#guard expectError (VerifierNormalForms.verifySNFCertificateJson missingSourceJson)
#guard expectError (VerifierNormalForms.verifySNFCertificateJson missingScalarJson)
#guard expectError (VerifierNormalForms.verifySNFCertificateJson unknownOpJson)
#guard expectError (VerifierNormalForms.verifySNFCertificateJson rowAddSelfJson)
#guard expectError (VerifierNormalForms.verifySNFCertificateJson colOutOfBoundsJson)
#guard expectError (VerifierNormalForms.verifySNFCertificateJson diagonalMismatchJson)
#guard expectError (VerifierNormalForms.verifySNFCertificateJson leftTransformMismatchJson)
#guard expectError (VerifierNormalForms.verifySNFCertificateJson rightTransformMismatchJson)
#guard expectError (VerifierNormalForms.verifySNFCertificateJson equationMismatchJson)
#guard expectError (VerifierNormalForms.verifySNFCertificateJson offDiagonalNonzeroJson)
#guard expectError (VerifierNormalForms.verifySNFCertificateJson divisibilityViolationJson)
#guard expectOk (VerifierNormalForms.verifySNFCertificateJson zeroRowJson)
#guard expectOk (VerifierNormalForms.verifySNFCertificateJson zeroColJson)
#guard !VerifierNormalForms.checkSNFCertificateJson diagonalMismatchJson

end Test.SNF
