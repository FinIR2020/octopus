import logging
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import time

logger = logging.getLogger(__name__)


class TaskCoordinatorAgent:

    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.active_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
    
    def coordinate_multi_domain_generation(
        self,
        tasks: List[Dict[str, Any]],
        generation_func,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Coordinate parallel data generation across multiple domains.
        
        Args:
            tasks: List of task dictionaries, each with:
                - 'domain': domain name
                - 'dataset': dataset name
                - 'data': training dataframe
                - 'config': task-specific configuration
            generation_func: Function to call for each task
            **kwargs: Additional arguments to pass to generation_func
            
        Returns:
            Dictionary mapping task_id -> generated dataframe
        """
        results = {}
        
        logger.info(f"Coordinating {len(tasks)} parallel generation tasks...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for i, task in enumerate(tasks):
                task_id = task.get('task_id', f"task_{i}")
                future = executor.submit(
                    self._execute_single_task,
                    task_id,
                    task,
                    generation_func,
                    **kwargs
                )
                future_to_task[future] = task_id
                self.active_tasks[task_id] = task
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[task_id] = result
                        self.completed_tasks[task_id] = self.active_tasks.pop(task_id)
                        logger.info(f"✓ Task {task_id} completed successfully")
                    else:
                        self.failed_tasks[task_id] = self.active_tasks.pop(task_id)
                        logger.warning(f"✗ Task {task_id} returned None")
                except Exception as e:
                    task_info = self.active_tasks.pop(task_id, {})
                    self.failed_tasks[task_id] = task_info
                    logger.error(f"✗ Task {task_id} failed: {e}")
        
        logger.info(f"Completed: {len(results)}/{len(tasks)} tasks successful")
        return results
    
    def _execute_single_task(
        self,
        task_id: str,
        task: Dict[str, Any],
        generation_func,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Execute a single generation task.
        
        Args:
            task_id: Unique task identifier
            task: Task configuration dictionary
            generation_func: Function to execute
            **kwargs: Additional arguments
            
        Returns:
            Generated dataframe or None if failed
        """
        try:
            start_time = time.time()
            logger.info(f"Starting task {task_id} (domain: {task.get('domain', 'unknown')})")
            
            # Prepare task-specific arguments
            task_kwargs = {**kwargs, **task.get('config', {})}
            
            # Execute generation
            result = generation_func(
                dataset=task.get('dataset', 'unknown'),
                data=task.get('data'),
                domain=task.get('domain'),
                **task_kwargs
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Task {task_id} completed in {elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in task {task_id}: {e}")
            return None
    
    def aggregate_results(
        self,
        results: Dict[str, pd.DataFrame],
        strategy: str = "concatenate"
    ) -> pd.DataFrame:
        """
        Aggregate results from multiple tasks.
        
        Args:
            results: Dictionary of task_id -> dataframe
            strategy: Aggregation strategy ('concatenate', 'union', 'intersection')
            
        Returns:
            Aggregated dataframe
        """
        if not results:
            return pd.DataFrame()
        
        if strategy == "concatenate":
            # Simple concatenation
            dfs = list(results.values())
            aggregated = pd.concat(dfs, ignore_index=True)
            logger.info(f"Aggregated {len(results)} results into {len(aggregated)} rows")
            return aggregated
        
        elif strategy == "union":
            # Union of all columns
            all_cols = set()
            for df in results.values():
                all_cols.update(df.columns)
            
            dfs_aligned = []
            for df in results.values():
                df_aligned = df.reindex(columns=list(all_cols))
                dfs_aligned.append(df_aligned)
            
            aggregated = pd.concat(dfs_aligned, ignore_index=True)
            return aggregated
        
        elif strategy == "intersection":
            # Intersection of columns
            if not results:
                return pd.DataFrame()
            
            common_cols = set(list(results.values())[0].columns)
            for df in list(results.values())[1:]:
                common_cols &= set(df.columns)
            
            dfs_aligned = []
            for df in results.values():
                df_aligned = df[list(common_cols)]
                dfs_aligned.append(df_aligned)
            
            aggregated = pd.concat(dfs_aligned, ignore_index=True)
            return aggregated
        
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")
    
    def get_task_status(self) -> Dict[str, Any]:
        """
        Get status of all tasks.
        
        Returns:
            Status dictionary with counts and details
        """
        return {
            "active": len(self.active_tasks),
            "completed": len(self.completed_tasks),
            "failed": len(self.failed_tasks),
            "total": len(self.active_tasks) + len(self.completed_tasks) + len(self.failed_tasks),
            "active_tasks": list(self.active_tasks.keys()),
            "completed_tasks": list(self.completed_tasks.keys()),
            "failed_tasks": list(self.failed_tasks.keys()),
        }
