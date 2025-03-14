// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { WriteStream } from 'fs';
import { bootstrap as globalAgentBootstrap } from 'global-agent';
import * as https from 'https';
import { JSZipObject } from 'jszip';

// Bootstrap global-agent to honor the proxy settings in
// environment variables, e.g. GLOBAL_AGENT_HTTPS_PROXY.
// See https://github.com/gajus/global-agent/blob/v3.0.0/README.md#environment-variables for details.
globalAgentBootstrap();

export const downloadZip = async (url: string): Promise<Buffer> =>
  new Promise<Buffer>((resolve, reject) => {
    https.get(url, (res) => {
      const { statusCode } = res;
      const contentType = res.headers['content-type'];

      if (statusCode === 301 || statusCode === 302) {
        downloadZip(res.headers.location!).then(
          (buffer) => resolve(buffer),
          (reason) => reject(reason),
        );
        return;
      } else if (statusCode !== 200) {
        throw new Error(`Failed to download build list. HTTP status code = ${statusCode}`);
      }
      if (!contentType || !/^application\/zip/.test(contentType)) {
        throw new Error(`unexpected content type: ${contentType}`);
      }

      const chunks: Buffer[] = [];
      res.on('data', (chunk) => {
        chunks.push(chunk);
      });
      res.on('end', () => {
        resolve(Buffer.concat(chunks));
      });
      res.on('error', (err) => {
        reject(`${err}`);
      });
    });
  });

export const extractFile = async (entry: JSZipObject, ostream: WriteStream): Promise<void> =>
  new Promise<void>((resolve, reject) => {
    entry
      .nodeStream()
      .pipe(ostream)
      .on('finish', () => {
        resolve();
      })
      .on('error', (err) => {
        reject(err);
      });
  });
