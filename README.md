# Multi-arm coordination for foraging

Controlling multiple arms to perform complex tasks is challenging especially when planning with limited information is involved. One example for an octopus robot is foraging food using multiple arms.
![foraging-problem.jpeg](foraging-problem.jpeg)

I place an octopus robot in an environment consists of random food targets, and it can only sense a few food targets nearby. The foraging goal is to gained as much energy as it can to survive. The octopus robot needs to consider the energy from food intake and the muscle energy cost. This task requires (1) controlling muscle activations for reaching for food ([Details of the muscle-activated soft arm model](https://github.com/chshih2/Real-time-control-of-an-octopus-arm-NNES)) (2) long-term planning on whether to reach for food or to coordinate arms to relocate to a new, more favorable position. 

## Hierarchical decomposition control
To solve this problem, I employ a hierarchical framework Inspired by the highly distributed neural system in an octopus. An octopus has a highly distributed neural system that comprises a Central Nervous System and a Peripheral Nervous System. While the central nervous system is responsible for learning and decision making by integrating signals from the entire body, the Peripheral Nervous System is responsible for locally control the muscles.
![hierarchical-control.gif](hierarchical-control.gif)


Based on this, a two-level hierarchical framework can be constructed. At the central-level, strategic commands are issued to the individual arms. Then, at the arm-level, embedded motor programs are executed to complete the selected commands. 

![decomposition.gif](decomposition.gif)

To solve the foraging problem for the soft octopus robot, I employ [NN-ES](https://github.com/chshih2/Real-time-control-of-an-octopus-arm-NNES) as the fast-responding arm-level controller for muscle activations. This enables the central-level controller to solve the foraging problem with sequences of crawl and reach. Given the crawl command, a prescribed crawling muscle activation is applied. If reach is selected, the arm attempts to reach all food targets within its workspace. 

This high-level control problem is solved using Proximal Policy Optimization (PPO)[1]. Here is the demonstration of a single arm using the framework. The strategy here reflects the consideration of the muscle energy cost, in the sense that the octopus will crawl until a large number of food targets are within reach, to then fetch them all at once. 

![1arm.gif](1arm.gif)

I evaluate the performance of PPO against four other approaches. The end-to-end approach has to learn the activations from scratch without leveraging the intermediate command. The other three approaches have hierarchical structure with different high-level controllers. A random controller selects crawl or reach with 50/50 probability. A ‘greedy’ controller selects reach whenever food is available, otherwise it selects crawl. Q is a policy derived from a simplified problem with assumptions on the full-knowledge of the targets, simplified workspace shape and the constant cost regardless of the muscle activations.

![decomposition_performance.jpeg](decomposition_performance.jpeg)
I evaluate their performance on the total gained energy, and average number of time steps per food target collected. Overall, all four hierarchical policies (including the random policy) outperform the end-to-end approach. In addition, among the hierarchical policies, the PPO policy not only collects food with less muscle energy costs, but also in a fewer number of steps.

## Multi-arm coordination
For foraging with multiple arms, I formulate the multi-arm problem in two ways: a centralized approach and a decentralized approach inspired by the two hypothesis in octopus control. The centralized approach is rooted in the hypothesis that the CNS coordinates all high-level decision-making, so one agent controls the actions of all arms. For the decentralized approach, it is based on that individual arms have been shown to possess a degree of independent decision-making. I explore this paradigm by formulating the high-level control as a multi-agent problem where each arm acts independently while contributing to learning via a shared-network policy. 

![centralized-decentralized.jpeg](centralized-decentralized.jpeg)

For foraging with two arms, within the same training episodes, the decentralized approach outperforms the alternative high-level controllers with more gained energy and fewer steps, while the analytical solution Q is comparable to the centralized approach.
![2arm.jpeg](2arm.jpeg)

Finally, I implement the framework for foraging with four arms. At this stage, the analytical solution Q is no longer available, this highlights the use of learning-based approaches. On top of that, this time the environment has randomly generated obstacles. I employ the [sensory reflex](https://github.com/chshih2/Real-time-control-of-an-octopus-arm-NNES) as the local-level controller to tackle this situation. The octopus crawls around to explore the arena and when the arm detects targets, it will try to collect.
![4arm.gif](4arm.gif)

Both centralized and decentralized approaches successfully learn to forage when training without the presence of obstacles, resulting in  collecting on average 88% and 99% of the food. I then deploy both policied in an environment littered with unmovable obstacles. Without further training, this leads the arms to arms becoming stuck, substantially impairing foraging behavior, which now only achieves 21% (centralized) and 25% (decentralized) food collection. With the [sensory reflex](https://github.com/chshih2/Real-time-control-of-an-octopus-arm-NNES), however, the octopus robot successfully collects 67% (centralized) and 75% (decentralized) of the food in an adverse environment.
![4arm_rate.jpeg](4arm_rate.jpeg)



[1] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
