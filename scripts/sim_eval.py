from absl import app

from orca.sim.widowx_sim_env import WidowXSimEnv
from orca.utils.run_eval import run_eval_loop

if __name__ == "__main__":

    def main(_):
        env = WidowXSimEnv(image_size=256)

        # this is the function that will be called to initialize the goal
        # condition to get the observation
        def get_goal_condition():
            return env.get_observation()

        # run the evaluation loop
        run_eval_loop(env, get_goal_condition, 0.1)

    app.run(main)
