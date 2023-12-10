import warnings

from clearml import Task
from clearml.automation import (
    HyperParameterOptimizer,
    DiscreteParameterRange,
    UniformParameterRange,
    UniformIntegerParameterRange,
    LogUniformParameterRange
)
from clearml.automation.optuna import OptimizerOptuna


warnings.filterwarnings('ignore')


def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('Objective reached {}'.format(objective_value))


task = Task.init(
    project_name='HPO',
    task_name='HP optimization',
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)

args = {
    'template_task_id': None,
}
args = task.connect(args)

if not args['template_task_id']:
    args['template_task_id'] = Task.get_task(
        project_name='HPO', task_name='HP optimization base').id


optimizer = HyperParameterOptimizer(
    base_task_id=args['template_task_id'],
    hyper_parameters=[
        UniformIntegerParameterRange('General/lin1_size', min_value=64, max_value=128, step_size=16),
        UniformIntegerParameterRange('General/lin2_size', min_value=32, max_value=64, step_size=16),
        UniformParameterRange('General/p_dropout', min_value=0.0, max_value=0.4, step_size=0.1),
        LogUniformParameterRange('General/lr', min_value=-3, max_value=-1),
        DiscreteParameterRange('General/batch_size', values=[2, 4, 8]),
    ],
    objective_metric_title='Validation Accuracy',
    objective_metric_series='accuracy',
    objective_metric_sign='max',
    optimizer_class=OptimizerOptuna,
    max_number_of_concurrent_tasks=1,
    total_max_jobs=10,
    min_iteration_per_job=10,
    max_iteration_per_job=30,
    execution_queue='CPU,'
)

optimizer.set_report_period(0.5)
optimizer.start_locally(job_complete_callback=job_complete_callback)
optimizer.set_time_limit(in_minutes=40.0)
optimizer.wait()
top_exp = optimizer.get_top_experiments(top_k=3)
print([t.id for t in top_exp])
optimizer.stop()

print('Finished Optimization')
