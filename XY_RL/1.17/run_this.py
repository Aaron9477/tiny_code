# -*-coding:utf-8-*-

from maze_env import Maze
from RL_brain import DeepQNetwork
import utils

model_save = False

def run_maze():
    step = 0
    for episode in range(1000):
        # initial observation
        observation = env.reset()
        print("begin")

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)  # 存储当前状态信息

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":

    time_string = utils.get_string_time()
    print(time_string, " the test begins")

    # maze game
    env = Maze()

    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.98,   # 最终不再探索
                      replace_target_iter=300,
                      memory_size=4800,
                      e_greedy_origin=0.7,
                      e_greedy_increment = 0.0005,  # epsilon增长速度
                      model_load = False,
                      model_load_dir = "save/{time}/model.ckpt",
                      model_save_dir = "save/{time}/model.ckpt".format(time=time_string),
                      output_graph=False,
                      )
    env.after(100, run_maze)
    env.mainloop()
    if model_save:
        RL.model_saver()
    RL.plot_cost()