// automatically generated by the FlatBuffers compiler, do not modify

/* eslint-disable @typescript-eslint/no-unused-vars, @typescript-eslint/no-explicit-any, @typescript-eslint/no-non-null-assertion */

import * as flatbuffers from 'flatbuffers';

/**
 * deprecated: no longer using kernel def hashes
 */
export class DeprecatedNodeIndexAndKernelDefHash {
  bb: flatbuffers.ByteBuffer | null = null;
  bb_pos = 0;
  __init(i: number, bb: flatbuffers.ByteBuffer): DeprecatedNodeIndexAndKernelDefHash {
    this.bb_pos = i;
    this.bb = bb;
    return this;
  }

  static getRootAsDeprecatedNodeIndexAndKernelDefHash(
    bb: flatbuffers.ByteBuffer,
    obj?: DeprecatedNodeIndexAndKernelDefHash,
  ): DeprecatedNodeIndexAndKernelDefHash {
    return (obj || new DeprecatedNodeIndexAndKernelDefHash()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
  }

  static getSizePrefixedRootAsDeprecatedNodeIndexAndKernelDefHash(
    bb: flatbuffers.ByteBuffer,
    obj?: DeprecatedNodeIndexAndKernelDefHash,
  ): DeprecatedNodeIndexAndKernelDefHash {
    bb.setPosition(bb.position() + flatbuffers.SIZE_PREFIX_LENGTH);
    return (obj || new DeprecatedNodeIndexAndKernelDefHash()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
  }

  nodeIndex(): number {
    const offset = this.bb!.__offset(this.bb_pos, 4);
    return offset ? this.bb!.readUint32(this.bb_pos + offset) : 0;
  }

  kernelDefHash(): bigint {
    const offset = this.bb!.__offset(this.bb_pos, 6);
    return offset ? this.bb!.readUint64(this.bb_pos + offset) : BigInt('0');
  }

  static startDeprecatedNodeIndexAndKernelDefHash(builder: flatbuffers.Builder) {
    builder.startObject(2);
  }

  static addNodeIndex(builder: flatbuffers.Builder, nodeIndex: number) {
    builder.addFieldInt32(0, nodeIndex, 0);
  }

  static addKernelDefHash(builder: flatbuffers.Builder, kernelDefHash: bigint) {
    builder.addFieldInt64(1, kernelDefHash, BigInt('0'));
  }

  static endDeprecatedNodeIndexAndKernelDefHash(builder: flatbuffers.Builder): flatbuffers.Offset {
    const offset = builder.endObject();
    return offset;
  }

  static createDeprecatedNodeIndexAndKernelDefHash(
    builder: flatbuffers.Builder,
    nodeIndex: number,
    kernelDefHash: bigint,
  ): flatbuffers.Offset {
    DeprecatedNodeIndexAndKernelDefHash.startDeprecatedNodeIndexAndKernelDefHash(builder);
    DeprecatedNodeIndexAndKernelDefHash.addNodeIndex(builder, nodeIndex);
    DeprecatedNodeIndexAndKernelDefHash.addKernelDefHash(builder, kernelDefHash);
    return DeprecatedNodeIndexAndKernelDefHash.endDeprecatedNodeIndexAndKernelDefHash(builder);
  }
}
