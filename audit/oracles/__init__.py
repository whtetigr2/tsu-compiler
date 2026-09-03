"""audit/oracles -- independent reference implementations used to verify
LATTICE, per plan Task A4.

Nothing under this package may be imported by src/ or demo/ (see the plan's
File Structure table: "audit/ ... is documentation and verification code --
it must not be imported by src/ or demo/"). The reverse dependency -- this
package reading data produced by src/tsu (receipts, program.json, etc.) for
comparison -- is fine; what is forbidden is importing src/tsu's *code* into
an oracle that is supposed to be independent of it.
"""
