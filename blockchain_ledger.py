# blockchain_ledger.py
import hashlib
import json
from datetime import datetime

class Block:
    def __init__(self, index, model_hash, timestamp, previous_hash):
        self.index = index
        self.model_hash = model_hash
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

class ModelBlockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis = Block(0, "0", datetime.now().isoformat(), "0")
        self.chain.append(genesis)

    def add_model(self, model_path):
        with open(model_path, "rb") as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()
        new_block = Block(
            index=self.chain[-1].index + 1,
            model_hash=model_hash,
            timestamp=datetime.now().isoformat(),
            previous_hash=self.chain[-1].hash
        )
        self.chain.append(new_block)
        return new_block

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            prev = self.chain[i - 1]
            curr = self.chain[i]
            if curr.hash != curr.compute_hash():
                return False
            if curr.previous_hash != prev.hash:
                return False
        return True

    def save_chain(self, path="blockchain/chain.json"):
        os.makedirs("blockchain", exist_ok=True)
        with open(path, "w") as f:
            json.dump([b.__dict__ for b in self.chain], f, indent=2)