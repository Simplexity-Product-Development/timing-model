#!/usr/bin/env python
"""
This module is used to run a scheduling algorithm given resource constraints.
This script was designed to support an upcoming whitepaper.

Some structures of note:

ops = dictionary where key is name of operation and value is duration of
operation

conflicts = array of dimension N x M x M where N is the number of items and M is
the number of total operations. A conflict between operations is represented by
a nonzero value in the MxM matrix for a particular item.  For example, if M=3
operations, and operation 1 should not be started while operation 0 is running,
the corresponding conflict matrix for all items is

[
    [0, 0, 0],
    [1, 0, 0],
    [0, 0, 0]
]

status = array of dimension N x M where N is the number of items and M is the
number of total operations.  If element in status matrix is nonzero, that
corresponds to a particular operation running on a particular item.

schedule = array of dimension N x M where N is the number of items and M is the
number of total operations.  If element in status matrix is nonzero, that
corresponds to the start time of when a particular operation is run on a particular item.

"""

__author__ = "Neil Foxman"


import numpy as np

def op_idx(ops, op_name):
    """
    Get index of an operation from the operation name.
    """
    return list(ops).index(op_name)

def define_conflict(conflicts, dont_start_idx, while_running_idx, bidirectional=True):
    conflicts[dont_start_idx][while_running_idx] += 1
    if bidirectional:
        conflicts[while_running_idx][dont_start_idx] += 1
    return conflicts

def define_conflict_by_name(conflicts, ops, dont_start_name, while_running_name, bidirectional=True):
    dont_start_idx = op_idx(ops, dont_start_name)
    while_running_idx = op_idx(ops, while_running_name)
    return define_conflict(conflicts, dont_start_idx, while_running_idx, bidirectional)

def graph_conflicts(ops, conflicts):
    import graphviz
    g = graphviz.Graph()
    num_ops = len(ops)
    op_names = list(ops.keys())
    for idx, op_name in enumerate(op_names):
        g.node(op_name, f'op{idx}: {op_name}')

    for idx in range(num_ops):
        for jdx in range(num_ops):
            if conflicts[idx, jdx] > 0:
                if idx == jdx:
                    # Conflict with self
                    g.edge(op_names[idx], op_names[jdx], dir='both')
                elif conflicts[jdx, idx] > 0 and idx < jdx:
                    # Bidirectional conflict
                    g.edge(op_names[idx], op_names[jdx], dir='both')
                elif conflicts[jdx, idx] <= 0:
                    # Directional conflict
                    g.edge(op_names[idx], op_names[jdx], dir='forward')
    return g

def calc_schedule_duration(schedule, ops):
    # Get start time of last operation in the schedule
    last_op_start = np.max(schedule)

    # Get duration of last op (last operation in the ops list)
    last_op_duration = list(ops.values())[-1]

    # Calculate the total schedule duration
    duration = last_op_start + last_op_duration
    return duration

def calc_schedule_efficiency(schedule, ops):
    # Calculate optimum schedule duration by adding up all op durations
    min_duration = 0
    for op_durations in ops.values():
        min_duration += op_durations

    # Calculate actual schedule duration
    sched_duration = calc_schedule_duration(schedule, ops)

    return min_duration / sched_duration

def generate_sequential_ops(ops_single, conflicts_single, num_cycles):
    # Create new ops list assuming we run the same operations several times in a row
    ops = {}
    for cycle_idx in range(num_cycles):
        for op in ops_single.keys():
            ops[f'{op}_{cycle_idx}'] = ops_single[op]

    # Generate new conflicts matrix
    conflicts = np.tile(conflicts_single, (num_cycles, num_cycles))
    return (ops, conflicts)

def generate_A_B_wait_ops(ops_single, A_names, B_names, wait_names, num_cycles):
    # Determine durations of each A and B operation by collecting the specified operations in ops_single
    def collect_ops(names, ops_single):
        duration = 0
        for name in names:
            duration += ops_single[name]
        coll_name = ', '.join(names)
        return (duration, coll_name)
    (A_duration, A_str) = collect_ops(A_names, ops_single)
    (B_duration, B_str) = collect_ops(B_names, ops_single)
    (wait_duration, wait_str) = collect_ops(wait_names, ops_single)
    
    # Generate new ops list based on number of cycles
    ops = {}
    ops[f'{A_str}_0'] = A_duration
    for cycle in range(num_cycles):
        ops[f'{wait_str}_{cycle}'] = wait_duration
        if cycle < num_cycles - 1:
            ops[f'{B_str}_{cycle}/{A_str}_{cycle+1}'] = B_duration + A_duration
    ops[f'{B_str}_{num_cycles-1}'] = B_duration

    # Generate new conflicts matrix
    # Note that with A-B-wait operation lists, conflicts exist between every other operation
    num_ops = len(ops)
    conflicts = np.zeros([num_ops, num_ops])
    for i in range(0, num_ops, 2):
        for j in range(0, num_ops, 2):
            define_conflict(conflicts, i, j)

    return (ops, conflicts)


def run_scheduler(num_items, ops, conflicts, sync_starts=False):
    num_ops = len(ops)
    op_names = np.array(list(ops.keys()))
    op_durations = np.array(list(ops.values()))

    # Define tracker for which operation each item has gotten to
    item_op_cntr = np.zeros(num_items, dtype=int)

    # Define status matrix
    status = np.zeros([num_items, num_ops])

    # Placeholder for completed schedule
    schedule = np.zeros([num_items, num_ops])

    # Generate priority matrix
    priority = np.zeros([num_items, num_ops])
    for item_idx in range(num_items):
        for op_idx in range(num_ops):
            priority[item_idx][op_idx] = item_idx + op_idx * num_items
    # print(priority)

    # Run scheduling algorithm
    scheduling = True
    t = 0
    while scheduling:
        # print(f"~~~ Evaluating at t={t} ~~~")
        
        # Complete any item operations that are done
        # Get array indicating which items have running operation
        item_status = np.matmul(status, np.ones(num_ops))
        for item_idx in range(num_items):
            if item_status[item_idx] > 0:
                op_idx = item_op_cntr[item_idx] # Get operation that is running on this item
                op_duration = op_durations[op_idx] # Get Operation Duration
                start_time = schedule[item_idx][op_idx] # Get operation start time
                
                # Check if this operation is now complete
                if t >= start_time + op_duration:
                    # print(f"Completing operation '{op_names[op_idx]}' on item {item_idx}")
                    # Mark this item as no longer running an operation
                    status[item_idx][op_idx] = 0

                    # Increment the operation counter for this item
                    item_op_cntr[item_idx] += 1
        

        # In the case of synchronized start times, don't start any new ops if there are still some running.
        if sync_starts and np.max(status) > 0:
            starting_ops = False
        else:
            starting_ops = True
        # Start all ops that can be started on this time step starting with the highest priority
        while starting_ops:
            # Get array indicating which operations are running on any item
            op_status = np.matmul(status.transpose(), np.ones(num_items))

            # Get array indicating which items are running
            item_status = np.matmul(status, np.ones(num_ops))

            # Determine which operations would conflict with the running operations
            op_conflict_exists = np.matmul(conflicts, op_status)

            # Find highest priority item that can be started
            best_priority = -1
            item_to_start = -1
            for item_idx in range(num_items):
                # If no operation is running on this item and not all operations are complete
                if item_status[item_idx] <= 0 and item_op_cntr[item_idx] < num_ops:
                    next_op = item_op_cntr[item_idx]                    

                    # If the next operation will not compete with any running operation
                    if op_conflict_exists[next_op] <= 0:
                        # Save this item/operation if best priority we've seen so far
                        if best_priority < 0 or priority[item_idx][next_op] < best_priority:
                            best_priority = priority[item_idx][next_op]
                            item_to_start = item_idx
            
            # If a candidate to start is found
            if item_to_start >= 0:
                op_to_start = item_op_cntr[item_to_start]
                # print(f"Starting operation '{op_names[op_to_start]}' on item {item_to_start}")
                
                # Start the operation on that item
                status[item_to_start][op_to_start] = 1

                # log the start time for this operation on the schedule
                schedule[item_to_start][op_to_start] = t

            # Otherwise we are done searching for this time step
            else:
                starting_ops = False
        
        # If we have looked through all items and no operations running, we are done with scheduling
        if np.max(status) <= 0:
            # print(f"Scheduling complete!  Total time = {t}")
            scheduling = False
        
        # Otherwise increment time counter
        else:
            t += 1

    return schedule

def plot_schedule(schedule, op_names, fig_width=1400, fig_height=800, label_size=18, color_list=None):
    schedule_duration = calc_schedule_duration(schedule, op_names)


    print("Importing plotting tools...")
    import plotly.graph_objects as go
    import plotly.express as px

    print("Generating plot...", flush=True) # Flush ensures console is up to date before proceeding
    fig = go.Figure()
    for item_idx, item_sched in enumerate(schedule):
        for op_idx, op_start_time in enumerate(item_sched):
            t_start = op_start_time
            t_width = list(op_names.values())[op_idx]
            y_start = item_idx - 0.5
            y_height = 1

            if color_list is None:
                color_list = px.colors.qualitative.Plotly

            fig.add_shape(
                type="rect",
                x0=t_start,
                y0=y_start,
                
                x1=t_start+t_width,
                y1=y_start+y_height,
                line=dict(
                    color="black",
                    width=1,
                ),
                fillcolor=color_list[op_idx % len(color_list)],
            )

            fig.add_annotation(
                x=t_start+t_width/2,
                y=y_start+y_height/2,
                text=list(op_names.keys())[op_idx],
                font=dict(
                    # family="sans serif",
                    size=label_size,
                    color="black"
                ),
                showarrow=False,
                bgcolor='rgba(255,255,255,0.8)',
            )    

    # Clean up figure and axes
    fig.update_xaxes(range=[0, schedule_duration])
    fig.update_yaxes(range=[-0.5, schedule.shape[0]-0.5])
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Item Index",
        autosize=False,
        width=fig_width,
        height=fig_height,
        yaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = 1
        )
    )
    fig.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Run Scheduler on dummy data.',
    )
    parser.add_argument(
        '--num-items',
        action='store',
        default=10,
        type=int,
        help='Total number of items to schedule operations list on'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable plot of resulting schedule.'
    )
    args = parser.parse_args()
    # print(args)

    num_items = args.num_items

    # Define operations and their durations
    ops = {
        'wash':40,
        'peel':80,
        'cut':60,
        'boil':10*60,
        'dry':5*60,
        'mash':10*60,
    }
    num_ops = len(ops)
    
    # Define Conflicts between operations
    conflicts = np.zeros([num_ops, num_ops])

    define_conflict(conflicts, op_idx(ops, 'wash'), op_idx(ops, 'wash'))

    define_conflict(conflicts, op_idx(ops, 'peel'), op_idx(ops, 'wash'))
    define_conflict(conflicts, op_idx(ops, 'peel'), op_idx(ops, 'peel'))

    define_conflict(conflicts, op_idx(ops, 'cut'), op_idx(ops, 'wash'))
    define_conflict(conflicts, op_idx(ops, 'cut'), op_idx(ops, 'peel'))
    define_conflict(conflicts, op_idx(ops, 'cut'), op_idx(ops, 'cut'))

    define_conflict(conflicts, op_idx(ops, 'boil'), op_idx(ops, 'wash'))
    define_conflict(conflicts, op_idx(ops, 'boil'), op_idx(ops, 'peel'))
    define_conflict(conflicts, op_idx(ops, 'boil'), op_idx(ops, 'cut'))
    define_conflict(conflicts, op_idx(ops, 'boil'), op_idx(ops, 'boil'))

    define_conflict(conflicts, op_idx(ops, 'dry'), op_idx(ops, 'wash'))
    define_conflict(conflicts, op_idx(ops, 'dry'), op_idx(ops, 'peel'))
    define_conflict(conflicts, op_idx(ops, 'dry'), op_idx(ops, 'cut'))
    # define_conflict(conflicts, op_idx(ops, 'dry'), op_idx(ops, 'boil'))
    # define_conflict(conflicts, op_idx(ops, 'dry'), op_idx(ops, 'dry'))

    define_conflict(conflicts, op_idx(ops, 'mash'), op_idx(ops, 'wash'))
    define_conflict(conflicts, op_idx(ops, 'mash'), op_idx(ops, 'peel'))
    define_conflict(conflicts, op_idx(ops, 'mash'), op_idx(ops, 'cut'))
    define_conflict(conflicts, op_idx(ops, 'mash'), op_idx(ops, 'boil'))
    # define_conflict(conflicts, op_idx(ops, 'mash'), op_idx(ops, 'dry'))
    define_conflict(conflicts, op_idx(ops, 'mash'), op_idx(ops, 'mash'))

    # Run scheduler
    schedule = run_scheduler(num_items, ops, conflicts, sync_starts=False)

    schedule_duration = calc_schedule_duration(schedule, ops)
    print(f"Total Schedule duration is {schedule_duration}")

    schedule_efficiency = calc_schedule_efficiency(schedule, ops)
    print(f"Schedule efficiency is {schedule_efficiency:.2%}")

    if not args.no_plot:
        plot_schedule(schedule, ops)
