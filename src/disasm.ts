import type { ABI, ABIFunction, ABIEvent, StateMutability } from "./abi.js";
import type { OpCode } from "./opcodes.js";
import type { ProxyResolver } from "./proxies.js";

import { hexToBytes, bytesToHex, cborMapDecoder } from "./utils.js";

import { opcodes, pushWidth, isPush, isLog, isHalt, isCompare } from "./opcodes.js";

import { slotResolvers, SequenceWalletProxyResolver, FixedProxyResolver } from "./proxies.js";


function valueToOffset(value: Uint8Array): number {
    // FIXME: Should be a cleaner way to do this...
    return parseInt(bytesToHex(value), 16);
}

// BytecodeIter takes EVM bytecode and handles iterating over it with correct
// step widths, while tracking N buffer of previous offsets for indexed access.
// This is useful for checking against sequences of variable width
// instructions.
export class BytecodeIter {
    bytecode: Uint8Array;

    nextStep: number; // Instruction count
    nextPos: number; // Byte-wise instruction position (takes variable width into account)

    // TODO: Could improve the buffer by making it sparse tracking of only
    // variable-width (PUSH) instruction indices, this would allow relatively
    // efficient seeking to arbitrary positions after a full iter. Then again,
    // roughly 1/4 of instructions are PUSH, so maybe it doesn't help enough?

    posBuffer: number[]; // Buffer of positions
    posBufferSize: number;

    constructor(bytecode: string, config?: { bufferSize?: number }) {
        this.nextStep = 0;
        this.nextPos = 0;
        if (config === undefined) config = {};

        this.posBufferSize = Math.max(config.bufferSize || 1, 1);
        this.posBuffer = [];

        this.bytecode = hexToBytes(bytecode);
    }

    hasMore(): boolean {
        return (this.bytecode.length > this.nextPos);
    }

    next(): OpCode {
        if (this.bytecode.length <= this.nextPos) return opcodes.STOP;

        const instruction = this.bytecode[this.nextPos];
        const width = pushWidth(instruction);

        // TODO: Optimization: Could use a circular buffer
        if (this.posBuffer.length >= this.posBufferSize) this.posBuffer.shift();
        this.posBuffer.push(this.nextPos);

        this.nextStep += 1;
        this.nextPos += 1 + width;

        return instruction;
    }

    // step is the current instruction position that we've iterated over. If
    // iteration has not begun, then it's -1.
    step(): number {
        return this.nextStep - 1;
    }

    // pos is the byte offset of the current instruction we've iterated over.
    // If iteration has not begun then it's -1.
    pos(): number {
        if (this.posBuffer.length === 0) return -1;
        return this.posBuffer[this.posBuffer.length - 1];
    }

    // asPos returns an absolute position for a given position that could be relative.
    // Returns -1 if out of bounds.
    asPos(posOrRelativeStep: number): number {
        let pos = posOrRelativeStep;
        if (pos < 0) {
            pos = this.posBuffer[this.posBuffer.length + pos];
            if (pos === undefined) {
                return -1;
            }
        }
        return pos;
    }

    // at returns instruction at an absolute byte position or relative negative
    // buffered step offset. Buffered step offsets must be negative and start
    // at -1 (current step).
    at(posOrRelativeStep: number): OpCode {
        const pos = this.asPos(posOrRelativeStep);
        return this.bytecode[pos];
    }

    // value of last next-returned OpCode (should be a PUSHN intruction)
    value(): Uint8Array {
        return this.valueAt(-1);
    }

    // valueAt returns the variable width value for PUSH-like instructions (or
    // empty value otherwise), at pos pos can be a relative negative count for
    // relative buffered offset.
    valueAt(posOrRelativeStep: number): Uint8Array {
        const pos = this.asPos(posOrRelativeStep);
        const instruction = this.bytecode[pos];
        const width = pushWidth(instruction);
        return this.bytecode.slice(pos + 1, pos + 1 + width);
    }
}

// Opcodes that tell us something interesting about the function they're in
const interestingOpCodes : Set<OpCode> = new Set([
    opcodes.STOP, // No return value
    opcodes.RETURN, // Has return value?
    opcodes.CALLDATALOAD, // Has arguments
    opcodes.CALLDATASIZE, // FIXME: Is it superfluous to have these two?
    opcodes.CALLDATACOPY,
    opcodes.DELEGATECALL, // We use this to detect proxies
    opcodes.SLOAD, // Not pure
    opcodes.SSTORE, // Not view
    opcodes.REVERT,
    opcodes.CREATE, // Factory
    opcodes.CREATE2, // Factory
    // TODO: Add LOGs to track event emitters?
]);

export class Function {
    byteOffset: number; // JUMPDEST byte offset
    opTags: Set<OpCode>; // Track whether function uses interesting opcodes
    start: number; // JUMPDEST instruction offset
    jumps: Array<number>; // JUMPDEST instruction offsets this function can jump to
    end?: number; // Last instruction offset before the next JUMPDEST

    constructor(byteOffset: number = 0, start: number = 0) {
        this.byteOffset = byteOffset;
        this.start = start;
        this.opTags = new Set();
        this.jumps = [];
    }
}

export class Program {
    dests: { [key: number]: Function }; // instruction offset -> Function
    selectors: { [key: string]: number }; // function hash -> instruction offset
    notPayable: { [key: number]: number }; // instruction offset -> bytes offset
    fallback?: number; // instruction offset for fallback function

    eventCandidates: Set<string>; // PUSH32 found before a LOG instruction
    proxySlots: Array<string>; // PUSH32 found that match known proxy slots
    proxies: Array<ProxyResolver>;
    isFactory: boolean; // CREATE or CREATE2 detected
    metadata: Record<string, string>; // CBOR metadata at the end of the contract

    init?: Program; // Program embedded as init code

    constructor(init?: Program) {
        this.dests = {};
        this.selectors = {};
        this.notPayable = {};
        this.eventCandidates = new Set();
        this.proxySlots = [];
        this.proxies = [];
        this.isFactory = false;
        this.init = init;
    }
}

export function abiFromBytecode(bytecodeOrProgram: string|Program): ABI {
    const p = typeof bytecodeOrProgram === "string" ? disasm(bytecodeOrProgram) : bytecodeOrProgram;

    const abi: ABI = [];
    for (const [selector, offset] of Object.entries(p.selectors)) {
        // TODO: Optimization: If we only look at selectors in the jump table region, we shouldn't need to check JUMPDEST validity.
        if (!(offset in p.dests)) {
            // Selector does not point to a valid jumpdest. This should not happen.
            continue;
        }

        // Collapse tags for function call graph
        const fn = p.dests[offset];
        const tags = subtreeTags(fn, p.dests);

        const funcABI = {
            type: "function",
            selector: selector,
            payable: !p.notPayable[offset],
        } as ABIFunction;

        // Note that these are not very reliable because our tag detection
        // fails to follow dynamic jumps.
        let mutability : StateMutability = "nonpayable";
        if (funcABI.payable) {
            mutability = "payable";
        } else if (!tags.has(opcodes.SSTORE)) {
            mutability = "view";
        }
        // TODO: Can we make a claim about purity? Probably not reliably without handling dynamic jumps?
        // if (mutability === "view" && !tags.has(opcodes.SLOAD)) {
        //    mutability = "pure";
        // }

        funcABI.stateMutability = mutability;

        // Unfortunately we don't have better details about the type sizes, so we just return a dynamically-sized /shrug
        if (tags.has(opcodes.RETURN) || mutability === "view") {
            // FIXME: We assume outputs based on mutability, that's a hack.
            funcABI.outputs = [{type: "bytes", name: ""}];
        }
        if (tags.has(opcodes.CALLDATALOAD) || tags.has(opcodes.CALLDATASIZE) || tags.has(opcodes.CALLDATACOPY)) {
            funcABI.inputs = [{type: "bytes", name: ""}];
        }

        abi.push(funcABI);
    }

    for (const h of p.eventCandidates) {
        abi.push({
            type: "event",
            hash: h,
        } as ABIEvent);
    }

    return abi;
}

const _EmptyArray = new Uint8Array();

export function disasm(bytecode: string, config?: {onlyJumpTable: boolean}): Program {
    const { onlyJumpTable } = config || {};

    let p : Program = new Program();

    const selectorDests = new Set<number>();
    let invertedSelector = "";

    let lastPush32: Uint8Array = _EmptyArray;  // Track last push32 to find log topics
    let checkJumpTable: boolean = true;
    let resumeJumpTable = new Set<number>();
    let maxJumpDest = 0;
    let runtimeOffset = 0; // Non-zero if init deploy code is included
    let boundaryPos = -1;

    let currentFunction: Function = new Function();
    p.dests[0] = currentFunction;

    const code = new BytecodeIter(bytecode, { bufferSize: 5 });

    while (code.hasMore()) {
        const inst = code.next();
        const pos = code.pos();
        const step = code.step();

        // Track last PUSH32 to find LOG topics
        // This is probably not bullet proof but seems like a good starting point
        if (inst === opcodes.PUSH32) {
            const v = code.value();
            const resolver = slotResolvers[bytesToHex(v)];
            if (resolver !== undefined) {
                // While we're looking at PUSH32, let's find proxy slots
                p.proxies.push(resolver);
            } else {
                lastPush32 = v;
            }
            continue
        } else if (isLog(inst) && lastPush32.length > 0) {
            p.eventCandidates.add(bytesToHex(lastPush32));
            continue
        }

        // Possible minimal proxy pattern? EIP-1167
        if (inst === opcodes.DELEGATECALL &&
            code.at(-2) === opcodes.GAS) {

            if (isPush(code.at(-3))) {
                // Hardcoded delegate address
                // TODO: We can probably do more here to determine which kind? Do we care?
                const val = code.valueAt(-3);
                const addr = bytesToHex(
                    val.slice(val.length-20), // Might be padded with zeros
                    20,
                );
                // Check if proxy is already found. Rare to have more than 1, so we'll just O(N) here
                if (!p.proxies.find(p => p instanceof FixedProxyResolver && p.resolvedAddress === addr)) {
                    p.proxies.push(new FixedProxyResolver("HardcodedDelegateProxy", addr));
                }

            } else if (
                code.at(-3) === opcodes.SLOAD &&
                code.at(-4) === opcodes.ADDRESS
            ) {
                // SequenceWallet-style proxy (keyed on address)
                p.proxies.push(new SequenceWalletProxyResolver());
            }
        }

        // Find JUMPDEST labels
        if (inst === opcodes.JUMPDEST) {
            // End of the function, or disjoint function?
            if (isHalt(code.at(-2)) || code.at(-2) === opcodes.JUMP) {
                if (currentFunction) currentFunction.end = pos - 1 - runtimeOffset;
                currentFunction = new Function(step, pos);

                // We don't stop looking for jump tables until we find at least one selector
                if (checkJumpTable && Object.keys(p.selectors).length > 0) {
                    checkJumpTable = false;
                }
                if (resumeJumpTable.delete(pos)) {
                    // Continuation of a previous jump table?
                    // Selector branch trees start by pushing CALLDATALOAD or it was pushed before.
                    checkJumpTable = code.at(pos + 1) === opcodes.DUP1 || code.at(pos + 1) === opcodes.CALLDATALOAD;
                } else if (!checkJumpTable && resumeJumpTable.size === 0 && onlyJumpTable) {
                    // Exit early if we're only looking in the jump table
                    break;
                }
            } // Otherwise it's just a simple branch, we continue

            // Index jump destinations so we can check against them later
            p.dests[pos - runtimeOffset] = currentFunction;

            // Check whether a JUMPDEST has non-payable guards
            //
            // We look for a sequence of instructions that look like:
            // JUMPDEST CALLVALUE DUP1 ISZERO
            //
            // We can do direct positive indexing because we know that there
            // are no variable-width instructions in our sequence.
            if (
                code.at(pos + 1) === opcodes.CALLVALUE &&
                code.at(pos + 2) === opcodes.DUP1 &&
                code.at(pos + 3) === opcodes.ISZERO
            ) {
                p.notPayable[pos] = step;
                // TODO: Optimization: Could seek ahead 3 pos/count safely
            }

            // TODO: Check whether function has a simple return flow?
            // if (code.at(pos - 1) === opcodes.RETURN) { ... }

            continue;
        } else if (
            (isHalt(code.at(-2)) || code.at(-2) === opcodes.JUMP) &&
            runtimeOffset < pos &&
            maxJumpDest < pos
        ) {
            // Did we just find the end of the program?
            boundaryPos = pos;
            break;
        }

        // Annotate current function
        if (currentFunction.opTags !== undefined) {

            // Detect simple JUMP/JUMPI helper subroutines
            if ((inst === opcodes.JUMP || inst === opcodes.JUMPI) && isPush(code.at(-2))) {
                const jumpOffset = valueToOffset(code.valueAt(-2));
                currentFunction.jumps.push(jumpOffset - runtimeOffset);
                maxJumpDest = Math.max(maxJumpDest, jumpOffset);
            }

            // Tag current function with interesting opcodes (not including above)
            if (interestingOpCodes.has(inst)) {
                currentFunction.opTags.add(inst);

                // CREATE/CREATE2 are part of interestingOpCodes so we can leverage this short circuit
                if (inst === opcodes.CREATE || inst === opcodes.CREATE2) {
                    p.isFactory = true;
                }
            }
        }

        // Did we just copy code that might be the runtime code?
        // PUSH2 <RUNTIME OFFSET> PUSH1 0x00 CODECOPY
        if (code.at(-1) === opcodes.CODECOPY &&
            (   // Add 0x00 to stack
                code.at(-2) === opcodes.PUSH1 ||
                code.at(-2) === opcodes.PUSH0 ||
                code.at(-2) === opcodes.RETURNDATASIZE
            ) &&
            isPush(code.at(-3))
        ) {
            const offsetDest: number = valueToOffset(code.valueAt(-3));
            resumeJumpTable.add(offsetDest);
            runtimeOffset = offsetDest;
            continue;
        }

        if (pos === runtimeOffset &&
            currentFunction.opTags.has(opcodes.RETURN) &&
            !currentFunction.opTags.has(opcodes.CALLDATALOAD)
        ) {
            // Reset state, embed program as init
            p = new Program(p);
            currentFunction = new Function();
            p.dests[0] = currentFunction;
            checkJumpTable = true;
        }

        if (!checkJumpTable) continue; // Skip searching for function selectors at this point

        // We're in a jump table section now. Let's find some selectors.

        if (inst === opcodes.JUMP && isPush(code.at(-2))) {
            const offsetDest: number = valueToOffset(code.valueAt(-2));

            if (invertedSelector !== "") {
                p.selectors[invertedSelector] = offsetDest;
                selectorDests.add(offsetDest);
                invertedSelector = "";
            } else {
                // The table is continued elsewhere? Or could be a default target
                resumeJumpTable.add(offsetDest);
            }
        }

        // Beyond this, we're only looking with instruction sequences that end with 
        //   ... PUSHN <BYTEN> JUMPI
        if (!(
            code.at(-1) === opcodes.JUMPI && isPush(code.at(-2))
        )) continue;

        const offsetDest: number = valueToOffset(code.valueAt(-2));
        currentFunction.jumps.push(offsetDest);

        // Find callable function selectors:
        //
        // https://github.com/ethereum/solidity/blob/242096695fd3e08cc3ca3f0a7d2e06d09b5277bf/libsolidity/codegen/ContractCompiler.cpp#L333
        //
        // We're looking for a sequence of opcodes that looks like:
        //
        //    DUP1 PUSH4 0x2E64CEC1 EQ PUSH1 0x37    JUMPI
        //    DUP1 PUSH4 <SELECTOR> EQ PUSHN <OFFSET> JUMPI
        //    80   63    ^          14 60-7f ^       57
        //               Selector            Dest
        //
        // We can reliably skip checking for DUP1 if we're only searching
        // within `inJumpTable` range.
        //
        // Note that sizes of selectors and destinations can vary. Selector
        // PUSH can get optimized with zero-prefixes, all the way down to an
        // ISZERO routine (see next condition block).
        if (
            code.at(-3) === opcodes.EQ &&
            isPush(code.at(-4))
        ) {
            // Found a function selector sequence, save it to check against JUMPDEST table later
            let value = code.valueAt(-4);
            // 0-prefixed comparisons get optimized to a smaller width than PUSH4
            const selector: string = bytesToHex(value, 4);
            p.selectors[selector] = offsetDest;
            selectorDests.add(offsetDest);

            continue;
        }

        // Sometimes the positions get swapped with DUP2:
        //    PUSHN <SELECTOR> DUP2 EQ PUSHN <OFFSET> JUMPI
        if (
            code.at(-3) === opcodes.EQ &&
            code.at(-4) === opcodes.DUP2 &&
            isPush(code.at(-5))
        ) {
            // Found a function selector sequence, save it to check against JUMPDEST table later
            let value = code.valueAt(-5)
            // 0-prefixed comparisons get optimized to a smaller width than PUSH4
            const selector: string = bytesToHex(value, 4);
            p.selectors[selector] = offsetDest;
            selectorDests.add(offsetDest);

            continue;
        }

        // Sometimes the final selector is negated using SUB
        //    PUSHN <SELECTOR> SUB PUSHN <OFFSET> JUMPI ... <OFFSET> <JUMP>
        //                                        ^ We are here (at -1)
        if (
            code.at(-3) === opcodes.SUB &&
            isPush(code.at(-4))
        ) {
            // Found an inverted JUMPI, probably a final "else" condition
            invertedSelector = bytesToHex(code.valueAt(-4), 4);
            p.fallback = offsetDest; // Fallback
        }

        // In some cases, the sequence can get optimized such as for 0x00000000:
        //    DUP1 ISZERO PUSHN <OFFSET> JUMPI
        // FIXME: Need a better heuristic to descriminate the preceding value. This is hacky. :(
        if (
            code.at(-3) === opcodes.ISZERO &&
            code.at(-4) === opcodes.DUP1 &&
            ( code.at(-5) === opcodes.CALLDATASIZE ||
              code.at(-5) === opcodes.CALLDATALOAD ||
              code.at(-5) === opcodes.JUMPDEST ||
              code.at(-5) === opcodes.SHR
            )
        ) {
            const selector = "0x00000000";
            p.selectors[selector] = offsetDest;
            selectorDests.add(offsetDest);

            continue;
        }

        // Jumptable trees use GT/LT comparisons to branch jumps.
        //    DUP1 PUSHN <SELECTOR> GT/LT PUSHN <OFFSET> JUMPI
        if (
            code.at(-3) !== opcodes.EQ &&
            isCompare(code.at(-3)) &&
            code.at(-5) === opcodes.DUP1
        ) {
            resumeJumpTable.add(offsetDest);

            continue;
        }
    }

    let endBoundary : number|undefined = undefined;
    const finalByte = code.bytecode.slice(-1)[0];
    // Try to detect CBOR metadata (vs bytecode that does not include it)
    // https://playground.sourcify.dev/
    if (!isHalt(finalByte) || code.bytecode.slice(-2)[0] === 0) {
        const cborLength = valueToOffset(code.bytecode.slice(-2));
        endBoundary = -(cborLength + 2); // +4 for the length bytes
        const cborData = code.bytecode.slice(endBoundary).slice(0, -2);

        if (cborData.length === 51) {
            // We're going to cheat if we suspect it's {ipfs, solc}
            p.metadata = {
                ipfs: bytesToHex(cborData.slice(8, 42)),  // 2 bytes map + 6 bytes "ipfs" for 34 bytes value
                solc: bytesToHex(cborData.slice(48, 51)),  // 2 bytes map + 6 bytes "solc" for 3 bytes value
            }
        }
    }

    if ((boundaryPos > 0 && p.proxies.length === 0)) {
        // Slots could be stored outside of the program boundary and copied in
        // and we haven't found any proxy slots yet so let's check just in case...
        // This is unstructured data, so it could be anything. We can't parse it reliably.

        // We can skip the CBOR encoding based on length defined in the final byte
        const auxData = bytesToHex(code.bytecode.slice(boundaryPos, endBoundary));
        if (auxData.length >= 2 + 64) { // 0x is empty, plus at least enough for a slot
            // Look for known slots in extra data segment that could be CODECOPY'd
            for (const [slot, resolver] of Object.entries(slotResolvers)) {
                if (auxData.lastIndexOf(slot.slice(2)) === -1) continue;
                p.proxies.push(resolver);
            }
        }
    }

    return p;
}

function subtreeTags(entryFunc: Function, dests: { [key: number]: Function }): Set<OpCode> {
    let tags = new Set<OpCode>([]);
    const stack = new Array<Function>(entryFunc);
    const seen = new Set<number>();

    while (stack.length > 0) {
        const fn = stack.pop();
        if (!fn) continue;
        if (seen.has(fn.start)) continue;
        seen.add(fn.start);

        tags = new Set([...tags, ...fn.opTags]);
        stack.push(...fn.jumps.map(offset => dests[offset]))
    }
    return tags;
}
