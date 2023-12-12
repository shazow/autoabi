#!/usr/bin/env -S ts-node-script --esm

import { ethers } from "ethers";
import { whatsabi } from "../src/index.js";

const env = {
    INFURA_API_KEY: process.env.INFURA_API_KEY,
    ETHERSCAN_API_KEY: process.env.ETHERSCAN_API_KEY,
    PROVIDER: process.env.PROVIDER,
};
const provider = env.INFURA_API_KEY ? (new ethers.InfuraProvider("homestead", env.INFURA_API_KEY)) : ethers.getDefaultProvider("homestead");

// Helper
// https://stackoverflow.com/questions/11731072/dividing-an-array-by-filter-function 
const partitionBy = <T>(
  arr: T[],
  predicate: (v: T, i: number, ar: T[]) => boolean
) =>
  arr.reduce(
    (acc, item, index, array) => {
      acc[+!predicate(item, index, array)].push(item);
      return acc;
    },
    [[], []] as [T[], T[]]
  );

async function main() {
    const address = process.env["ADDRESS"] || process.argv[2];

    let r = await whatsabi.autoload(address, {
        provider,
        onProgress: (phase: string, ...args: string[]) => {
            console.debug("progress:", phase, ...args);
        },
        ... whatsabi.loaders.defaultsWithAPIKeys(env),
    });

    while (true) {
        const [abi, unresolved] = partitionBy(r.abi, a => (a.type !== "function" || "name" in a));
        const iface = new ethers.Interface(abi);
        console.log("autoload", iface.format());
        if (unresolved) console.log("unresolved", unresolved);

        if (!r.followProxies) break;

        console.log("following proxies...");
        r = await r.followProxies();
    }
}

main().then().catch(err => {
    console.error("Failed:", err)
    process.exit(2);
})
