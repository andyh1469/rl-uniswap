# A Reinforcement Learning Approach to Optimal Liquidity Provision in Uniswap v3

# Team Members
Andy Hsu, Akshay Sreekumar

# Description
This project simulates a decentralized liquidity pool (Uniswap v3) in Gym by formulating it as an MDP. We then use deep reinforcement learning to train liquidity provider agents in this environment. See [report.pdf](report.pdf) for more details.

# Run Instructions
First, unzip the blockchain data to get the required CSV files:
`unzip blockchain_data.zip`

Then simply run `train.py`. You can edit the input parameters in `config.yml`.
`python train.py`
