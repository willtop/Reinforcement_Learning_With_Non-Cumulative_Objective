# Reinforcement_Learning_With_Non-Cumulative_Objective

This code repository is for the TMLCN paper "Reinforcement Learning With Non-Cumulative Objective", available at https://ieeexplore.ieee.org/abstract/document/10151914.

For any reproduce, further research or development, please kindly cite our TMLCN Journal paper:
<pre>
@Article{non_cumulative, 
    author = "W. Cui and W. Yu", 
    title = "Reinforcement Learning With Non-Cumulative Objective", 
    journal = "{\it IEEE Trans. Mach. Learn. Commun. Netw.}", 
    year = 2023, 
    month = "June",
    note = "Early Access"
}
</pre>

In two folders, code files are included for running all simulations within the paper:
    In folder "Atari", the training and evaluation codes are provided for both CartPole and Atari Breakout.
    In folder "Adhoc_Networks", the training and evaluation codes are provided for routing in wireless ad hoc networks.
Please read the two specific README.txt files in both the "Atari" folder and the "Adhoc_Networks" folder for detailed descriptions on python scripts and stored neural network models that have been trained. 

Software requirements:
1. Standard Python libraries
2. numpy
3. pytorch
4. matplotlib
5. gym[atari]
6. pickle (for training and evaluation results saving and loading)
