# EdGym: OpenAI Gym Environments for Computational Education

Under construction...

Here are the instruction to install requirements for this code base.

```bash
pip install gym==0.15.7
```

List of supported environments:

1. Computer Adapative Test (CAT) Environment
2. Contextual Bandit Style AI Tutor Environment

You can run `cat_env.py` and `eye_env.py` right now. The implementation is also simple enough to understand and extend.

Note that currently both `CatEnv` and `EyeEnv` are defined for multiple students/patients.

`EyeEnv`'s Gumbel rejection sampling is very slow sometimes. Not sure how to fix it.
