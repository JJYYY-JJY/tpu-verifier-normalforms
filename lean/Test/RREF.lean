import VerifierNormalForms

namespace Test.RREF

def validJson : String :=
  "{\"kind\":\"rref_modp\",\"modulus\":5,\"shape\":[2,2],\"input\":[[1,2],[3,4]],\"ops\":[{\"kind\":\"add\",\"target\":1,\"source\":0,\"scalar\":-3},{\"kind\":\"scale\",\"target\":1,\"scalar\":2},{\"kind\":\"add\",\"target\":0,\"source\":1,\"scalar\":-2}],\"final\":[[1,0],[0,1]],\"pivots\":[{\"row\":0,\"col\":0},{\"row\":1,\"col\":1}]}"

def nonPrimeJson : String :=
  "{\"kind\":\"rref_modp\",\"modulus\":4,\"shape\":[1,1],\"input\":[[1]],\"ops\":[],\"final\":[[1]],\"pivots\":[{\"row\":0,\"col\":0}]}"

def badReplayJson : String :=
  "{\"kind\":\"rref_modp\",\"modulus\":5,\"shape\":[2,2],\"input\":[[1,2],[3,4]],\"ops\":[{\"kind\":\"add\",\"target\":1,\"source\":0,\"scalar\":-3},{\"kind\":\"scale\",\"target\":1,\"scalar\":2},{\"kind\":\"add\",\"target\":0,\"source\":1,\"scalar\":-2}],\"final\":[[1,0],[0,2]],\"pivots\":[{\"row\":0,\"col\":0},{\"row\":1,\"col\":1}]}"

def badPivotsJson : String :=
  "{\"kind\":\"rref_modp\",\"modulus\":5,\"shape\":[2,2],\"input\":[[1,2],[3,4]],\"ops\":[{\"kind\":\"add\",\"target\":1,\"source\":0,\"scalar\":-3},{\"kind\":\"scale\",\"target\":1,\"scalar\":2},{\"kind\":\"add\",\"target\":0,\"source\":1,\"scalar\":-2}],\"final\":[[1,0],[0,1]],\"pivots\":[{\"row\":0,\"col\":0}]}"

def missingPivotsJson : String :=
  "{\"kind\":\"rref_modp\",\"modulus\":5,\"shape\":[1,1],\"input\":[[1]],\"ops\":[],\"final\":[[1]]}"

def malformedOpJson : String :=
  "{\"kind\":\"rref_modp\",\"modulus\":5,\"shape\":[1,1],\"input\":[[1]],\"ops\":[{\"kind\":\"scale\",\"target\":0}],\"final\":[[1]],\"pivots\":[{\"row\":0,\"col\":0}]}"

def zeroScaleJson : String :=
  "{\"kind\":\"rref_modp\",\"modulus\":5,\"shape\":[1,1],\"input\":[[1]],\"ops\":[{\"kind\":\"scale\",\"target\":0,\"scalar\":5}],\"final\":[[1]],\"pivots\":[{\"row\":0,\"col\":0}]}"

def selfAddJson : String :=
  "{\"kind\":\"rref_modp\",\"modulus\":5,\"shape\":[1,1],\"input\":[[1]],\"ops\":[{\"kind\":\"add\",\"target\":0,\"source\":0,\"scalar\":1}],\"final\":[[2]],\"pivots\":[{\"row\":0,\"col\":0}]}"

def outOfBoundsOpJson : String :=
  "{\"kind\":\"rref_modp\",\"modulus\":5,\"shape\":[1,1],\"input\":[[1]],\"ops\":[{\"kind\":\"swap\",\"target\":0,\"source\":1}],\"final\":[[1]],\"pivots\":[{\"row\":0,\"col\":0}]}"

def expectOk : Except String Unit → Bool
  | .ok _ => true
  | .error _ => false

def expectError : Except String Unit → Bool
  | .ok _ => false
  | .error _ => true

#guard match VerifierNormalForms.parseRREFCertificateJson validJson with
  | .ok cert => VerifierNormalForms.checkRREFCertificate cert
  | .error _ => false

#guard expectOk (VerifierNormalForms.verifyRREFCertificateJson validJson)
#guard VerifierNormalForms.checkRREFCertificateJson validJson
#guard expectError (VerifierNormalForms.verifyRREFCertificateJson nonPrimeJson)
#guard expectError (VerifierNormalForms.verifyRREFCertificateJson badReplayJson)
#guard expectError (VerifierNormalForms.verifyRREFCertificateJson badPivotsJson)
#guard expectError (VerifierNormalForms.verifyRREFCertificateJson missingPivotsJson)
#guard expectError (VerifierNormalForms.verifyRREFCertificateJson malformedOpJson)
#guard expectError (VerifierNormalForms.verifyRREFCertificateJson zeroScaleJson)
#guard expectError (VerifierNormalForms.verifyRREFCertificateJson selfAddJson)
#guard expectError (VerifierNormalForms.verifyRREFCertificateJson outOfBoundsOpJson)
#guard !VerifierNormalForms.checkRREFCertificateJson badReplayJson

end Test.RREF
