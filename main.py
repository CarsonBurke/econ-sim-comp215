!pip install empiricaldist
from empiricaldist import Cdf, Pmf

import os
if not os.path.exists('utils.py'):
    !wget https://raw.githubusercontent.com/AllenDowney/ThinkComplexity2/master/notebooks/utils.py
if not os.path.exists('Cell2D.py'):
    !wget https://raw.githubusercontent.com/AllenDowney/ThinkComplexity2/master/notebooks/Cell2D.py


%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random as rand
import uuid

from utils import decorate

from scipy.signal import correlate2d
from Cell2D import Cell2D, draw_array
from functools import reduce
import operator
import sys
from pprint import pprint

MAX_AGENT_AGE = 100
# How much an agen't health decreases each tick
HEALTH_AGE_STEP = 0.0002
PROFESSIONS = ["research", "teaching", "farming", "building", "mining", "forestry", "business", "labour"]
# How likely the agent is to dropout of school based on how old it is
DROPOUT_CHANCE_PER_AGE = 0.0005
# The minimum age an agent must be before it drops out of school
MIN_DROPOUT_AGE = 6
# The learning rate for skills not related to the current profession, if no longer in school
NON_PROFESSION_LEARNING_RATE = 0.1
SCHOOL_LEARNING_RATE = 2
# How much better the agent is at their job based on their skill as a decaying rate - simulate decaying marginal ability
SKILL_JOB_ABILITY_RATE = 0.85
# Basic needs required to live (not required if still in school)
BASE_EXPENSES = 10_000
# Agents loose skill a small percent when they age
AGE_KNOWLEDGE_LOSS = 0.999
# Agents loose skill by an increasing amount with age
AGE_KNOWLEDGE_LOSS_AGE_MODIFIER = 0.0001
# How old the agent has to be before it can have kids
MIN_PARENT_AGE = 30
# Default chance of the agent wanting children
PARENT_CHANCE = 0.0005
# How demotivated the agent is from having children for each child it already has
ADDITIONAL_CHILD_DEMOTIVATION = PARENT_CHANCE * 20
# The minimum cost it takes to run a business per tick
MIN_BUSINESS_EXPENSE = 30_000
# The minimum employee salary
MIN_SALARY = 50_000
# The minimum percent of job postings to look through
MIN_JOB_SEARCHES = 5
# The minimum time an agent can be an employee before trying to transfer to a different business
MIN_EMPLOYMENT_TIME = 4
# How much a business makes per employee
PROFIT_PER_EMPLOYEE = 10_000

class EconSimulator(Cell2D):

  def __init__(self, n, start_agents_count):
    self.n = n

    self.create_agents(start_agents_count)
    self.society = Society()
    # Businesses mapped by their id
    self.businesses = {}
    self.history = History()

  def create_agents(self, start_agents_count):

    positions = [i for i in range(self.n)]
    rand.shuffle(positions)

    # Create agents with random starting wealth amounts
    self.agents = {}

    for i in positions[0:start_agents_count]:
      agent = Agent(i, 100_000)
      self.agents[agent.id] = agent

  def run(self):
    for i in range(self.n):
      self.step(i)

  def step(self, tick):

    taxes_paid = 0

    # Track who is working what jobs

    professions = create_work_ledger()

    for agent in self.agents.values():
      if agent.profession == None:
        continue

      professions[agent.profession] += 1

    # Try to start businesses

    for agent in self.agents.values():
      if agent.can_start_business() == False:
        continue

      # Start a business

      business = Business(agent.id)
      self.businesses[business.id] = business
      agent.business_id = business.id

      agent.wealth -= MIN_BUSINESS_EXPENSE
      business.assets += MIN_BUSINESS_EXPENSE

      print(f"Starting a new business with id {business.id}")

    # Either go to school or try to work

    for agent in self.agents.values():
      if agent.in_school:
        agent.school(self.society)
        continue

      income = agent.try_work(self.society, self.businesses)
      # If they made money, tax them
      if income:
        tax = income * self.society.tax_rate

        agent.wealth -= tax
        taxes_paid += tax

    # Run businesses

    agent_ids_set = set(self.agents.keys())

    business_ids = list(self.businesses.keys())
    for business_id in business_ids:
      business = self.businesses[business_id]

      # If the owner is dead, delete the business
      if not business.owner_id in agent_ids_set:
        del self.businesses[business.id]
        continue

      owner = self.agents[business.owner_id]

      if owner.run_business(business, self.agents):
        continue

      # The business has failed, liquidate the business

      owner.wealth += business.assets * 0.1

      del self.businesses[business.id]
      owner.business_id = None

      print(f"Deleted business with id {business.id}")

    # Age and expenses

    agent_keys = list(self.agents.keys())
    for agent_id in agent_keys:
      agent = self.agents[agent_id]

      # If the agent died from old age
      if agent.step_age() == False:
        del self.agents[agent.id]
        continue

      # If we are not in school, we have to pay for expenses
      if agent.in_school == False:
        # If we died due to not having money to pay for expenses
        if agent.expenses() == False:
          del self.agents[agent.id]
      # If we are in school, society pays for basic needs
      else:
        self.society.funds -= BASE_EXPENSES

    # Children

    agent_keys = list(self.agents.keys())
    for agent_id in agent_keys:
      agent = self.agents[agent_id]

      if agent.wants_kids() == False:
        continue

      inheritance = agent.wealth / 2

      new_agent = Agent(agent.pos, inheritance)
      self.agents[new_agent.id] = new_agent

      agent.wealth -= inheritance
      agent.children += 1

      print(f"Created child")

    # Society updates

    self.society.funds += taxes_paid
    self.society.update_desires()
    self.society.update_tax_rate()

    # Extra History updates
    print(f"recording history; remaining agents: {len(self.agents)} tick: {tick}")

    self.history.tax_income.append(self.society.tax_rate)
    self.history.gov_funds.append(self.society.funds)

    self.history.agent_count.append(len(self.agents))

    total_assets = reduce(lambda sum, agent_id: sum + self.agents[agent_id].wealth, self.agents, 0)
    self.history.total_assets.append(total_assets)
    self.history.avg_assets.append(total_assets / (len(self.agents) + sys.float_info.epsilon))

    agent_skills = map(lambda agent_id: reduce(operator.add, self.agents[agent_id].skills.values()), self.agents)

    total_skill = reduce(lambda sum, skills_sum: sum + skills_sum, list(agent_skills), 0)
    avg_skill = total_skill / (len(self.agents) + sys.float_info.epsilon)
    self.history.avg_skill.append(avg_skill)

    self.history.businesses_count.append(len(self.businesses))

    in_school_count = reduce(lambda sum, agent_id: sum + 1 if self.agents[agent_id].in_school else 0, self.agents, 0)
    print(f"in school {in_school_count}")
    self.history.in_school.append(in_school_count)
    # Workers are those who are not in school or running businesses
    self.history.at_work.append(len(self.agents) - in_school_count - len(self.businesses))

  def draw(self):
    """Draws the cells."""
    draw_array(self.array, cmap='YlOrRd', vmax=9, origin='lower')

    # draw the agents
    xs, ys = self.get_coords()
    self.points = plt.plot(xs, ys, '.', color='red')[0]

def get_coords(self):
    """Gets the coordinates of the agents.

    Transforms from (row, col) to (x, y).

    returns: tuple of sequences, (xs, ys)
    """
    agents = self.agents
    rows, cols = np.transpose([agent.pos for agent in agents])
    xs = cols + 0.5
    ys = rows + 0.5
    return xs, ys

class Society():
  def __init__(self):
    # How much each skill is socially desired
    self.desired_skills = create_work_ledger()
    # How much unspent tax money has accumilated (can be negative as a deficit)
    self.funds = 0
    # How much to tax income-earners on
    self.tax_rate = 0.1

  def update_desires(self):
    """Randomly change desired skills to simulate changes in culture and material needs"""
    for skill in self.desired_skills:
      self.desired_skills[skill] = min(max(self.desired_skills[skill] + rand.uniform(-1, 1), 0.1), 2)

  def update_tax_rate(self):
    if self.funds <= 0:
      self.tax_rate += 0.0005
    else: self.tax_rate -= 0.0005

    self.tax_rate = max(min(self.tax_rate, 0.5), 0)

class Business():
  def __init__(self, owner_id):
    self.id = uuid.uuid4()
    self.assets = 0
    self.owner_id = owner_id
    self.employee_ids = []
    self.salaries = create_work_ledger_random()

class History():
  def __init__(self):
    self.tax_income = []
    self.gov_funds = []
    self.agent_count = []
    self.total_assets = []
    self.avg_assets = []
    self.avg_skill = []
    self.businesses_count = []
    self.in_school = []
    self.at_work = []

  def graph(self):
    """
    Graph all of the below-specified graph subjects of history
    """
    graph_subjects = {
        "tax rate": self.tax_income,
        "gov' funds": self.gov_funds,
        "agent count": self.agent_count,
        "total assets": self.total_assets,
        "average assets": self.avg_assets,
        "average skill": self.avg_skill,
        "business count": self.businesses_count,
        "in school": self.in_school,
        "at work": self.at_work,
    }
    graph_keys = list(graph_subjects.keys())
    subjects_count = len(graph_subjects)

    fig, axs = plt.subplots(subjects_count, 1, figsize=(8, 4 * subjects_count))

    for index in range(subjects_count):

      graph_title = graph_keys[index]

      graph_data = graph_subjects[graph_title]

      axs[index].plot(range(len(graph_data)), graph_data)
      axs[index].set_title(graph_title)
      
# Motivations from 0-1
class Motivations():
  def __init__(self):
    self.financial = 0
    self.social = 0

# Can't use a class since we need to index the keys dynamically
# Skills 0-1
def create_work_ledger():
  return {
      "research": 0,
      "teaching": 0,
      "farming": 0,
      "building": 0,
      "mining": 0,
      "forestry": 0,
      "business": 0,
      "labour": 0,
  }

def create_work_ledger_random():
  ledger = create_work_ledger()

  for skill in ledger:
      ledger[skill] = rand.random()

  return ledger

class Agent():
  def __init__(self, pos, inherit_wealth, inherited_skills = create_work_ledger_random()):
    self.id = uuid.uuid4()
    self.health_death_chance = 0
    self.age = 0
    self.wealth = inherit_wealth
    self.children = 0
    # The id of the business the agent owns
    self.business_id = None
    self.pos = pos
    self.in_school = True
    # The id of the business that employs the agent
    self.employment_id = None
    self.employment_time = 0
    self.profession = None
    self.motivations = Motivations()
    self.inherit_skills(inherited_skills)

  def inherit_skills(self, inherited_skills):
    """
    Inherit skills from the parent based relative to their highest skill

    The idea is to simulate learning from parents and tedning to want to do the work they do
    """

    # Give the agent random skill
    self.skills = create_work_ledger_random()

    highest_inherited = max(inherited_skills.values())

    # Then inherit the parent's
    for skill in inherited_skills:
      self.skills[skill] += inherited_skills[skill] / highest_inherited

  def step_age(self) -> False:
    """Returns: bool: if the agent survived the aging step"""

    # Chance to die based on age-related factors
    if rand.random() < self.health_death_chance: return False

    # How motivated the agent is to live - simulates eating healthier, putting on sunscreen, etc.
    motivation = self.motivations.financial + self.motivations.social

    health_chance_delta = HEALTH_AGE_STEP - (HEALTH_AGE_STEP * motivation)
    self.health_death_chance += health_chance_delta

    # Loose skill knowledge faster as the agent gets older

    for skill in self.skills:
      skill_delta = AGE_KNOWLEDGE_LOSS + (AGE_KNOWLEDGE_LOSS_AGE_MODIFIER * self.age)
      self.skills[skill] *= skill_delta

    self.age += 1
    return True

  def school(self, society):
    """Have the agent go to school"""
    # Will get better at skills based on:
    # parent's profession, modified by social
    # society's desires, modified by social
    # financial prospect, modified by financial

    # Could use reduce but am lazy

    total_skill = 0

    for skill in self.skills:
      total_skill += self.skills[skill]

    avg_skill = total_skill / len(self.skills)

    # Increase all skills based on motivation
    # Increase skills that we are already good at - specialization

    for skill in self.skills:
      # Above-average skill increase to simulate specialization
      distance = min(max((self.skills[skill] - avg_skill), 0), 0.1)

      desire = self.motivations.social * (1 + society.desired_skills[skill]) + self.motivations.financial + distance
      skill_delta = desire * SCHOOL_LEARNING_RATE
      self.skills[skill] += skill_delta

    # Can't drop out of school until we get old enough

    if self.age < MIN_DROPOUT_AGE:
      return

    # Chance to drop out of school, encouraged by lack of motivation

    motivation = self.motivations.financial + self.motivations.social
    dropout_chance = (1 + motivation) - (DROPOUT_CHANCE_PER_AGE * self.age)

    if rand.random() > dropout_chance:
      self.in_school = False
      print(f"agent dropping out of school: {self.id}")
      return

  def try_work(self, society, businesses) -> float | None:
    """
    Try to perform a job if we meet conditions, such as not owning a business and having/finding a job to do
    """
    # If we are a business owner, stop

    business_owner_ids = set(map(lambda b_id: businesses[b_id].owner_id, businesses))
    if self.id in business_owner_ids:
      return

    # If we are registered as being employed and the employer (business) exists, work the job
    if self.employment_id and businesses[self.employment_id]:
      if self.try_update_job(businesses):
        return

      return self.job()

    # If we don't have an employer we need to find a job

    if self.choose_and_find_job(businesses):
      return self.job(society)

  def choose_and_find_job(self, businesses) -> bool:
    """
    Finds and becomes an employee of the best job the agent is motivated to find

    Returns: Wether or not a job was found
    """
    job = self.find_job(businesses)
    if job == None:
      return False

    # Assign the employee and business to each other

    business = businesses[job[2]]
    skill = job[0]
    self.choose_job(business, skill)

    return True

  def try_update_job(self, businesses) -> bool:
    """
    So long as we pass the requirements, try to pick a job with better pay
    """

    # If we haven't been employed at the current job for long enough, stop
    if self.employment_time < MIN_EMPLOYMENT_TIME:
      return False

    current_business = businesses[self.business_id]
    new_job = self.find_job()

    # If the new job pays worse, stop
    if new_job[1] <= self.salary:
      return False

    return True

  def choose_job(self, business, skill):
    business.employee_ids.append(self.id)
    self.business_id = business.id
    self.employment_time = 0
    self.profession = skill

  def find_job(self, businesses):
    """
    Find the best job of a list of potential jobs

    Returns: The skill, salary, and business id of the job
    """
    # Special care for those who have high business skill

    # Find a job from a local business
    # The further away the business, the more expensive it is to move there

    # Construct a list of jobs

    # Each business is considered to be offering one job
    jobs_count = len(businesses)
    # List of (skill, salary, business_id)
    jobs = []

    for business_id in businesses:
      business = businesses[business_id]

      for skill in business.salaries:
        salary = business.salaries[skill]
        jobs.append((skill, salary, business.id))

    if len(jobs) == 0:
      return None

    # Based on motivation, search through some of the jobs and choose the best paying one

    rand.shuffle(jobs)

    job = max(jobs, key=lambda job: job[1] * (1 + self.skills[job[0]]))
    return job

  def job(self, society) -> float:
    """
    Have the agent work its job

    Returns: how much the the job paid
    """
    # Will make money based on:
    # societal demand for the job
    # skill in the job

    # Work the job

    job_ability = self.skills[self.profession] ** SKILL_JOB_ABILITY_RATE
    pay = MIN_SALARY * job_ability * society.desired_skills[self.profession]

    self.wealth += pay

    # Improve the agent's skill based on their motivations

    skill_delta = self.motivations.financial + self.motivations.social
    self.skills[self.profession] += skill_delta

    # Slightly increase other skills based on motivation
    # Simulate general knowledge gain with age

    for skill in self.skills:
      skill_delta = self.motivations.financial + self.motivations.social * NON_PROFESSION_LEARNING_RATE
      self.skills[skill] += skill_delta

    return pay

  def expenses(self) -> False:
    """
    Simulate food, shelter, water, etc. costs, as well as conditional and correlary luxury-goods expenses

    Returns: If the agent survived paying their expenses
    """
    # Basic costs like food, water, housing, etc.
    self.wealth -= BASE_EXPENSES

    # If we could not afford basic expenses, we die
    if self.wealth < 0:
      print(f"Insufficient funds to pay base expenses {self.id}")
      return False

    # Additional expenses on "luxury goods" based on motivations

    net_motivation = self.motivations.social - self.motivations.financial
    if net_motivation > 0:
      # Will only spend a portion of wealth
      additional_expenses = (self.wealth / 8) * net_motivation
      self.wealth -= additional_expenses

    return True

  def wants_kids(self) -> bool:
    """Returns: If the agent wants to have a child"""

    if self.age < MIN_PARENT_AGE: return False
    if self.in_school: return False

    # Simulate the financial disincentive to have a child vs social desire to
    net_motivation = max(self.motivations.social - self.motivations.financial, 0.1)

    # Default chance to have a child based on age, take away chance based on additional children
    child_chance = (PARENT_CHANCE * self.age) - (ADDITIONAL_CHILD_DEMOTIVATION * self.children)
    return rand.random() < child_chance

  def run_business(self, business, agents) -> bool:
    """Returns: If the business is still opperational"""

    # Expenses and assets conversion

    expenses = MIN_BUSINESS_EXPENSE
    self.wealth -= expenses
    # Some of the expenses are converted into liquid assets
    business.assets += expenses * 0.1

    # Profit

    profit = 0

    for employee_id in business.employee_ids:
      try:
        employee = agents[employee_id]
      except:
        continue

      profit += PROFIT_PER_EMPLOYEE * (1 + self.motivations.financial - self.motivations.social)
    print(f"business profit {profit}")
    self.wealth += profit

    if self.wealth < 0: return False
    return True

  def can_start_business(self) -> bool:
    """
    See if we meet the conditions to start a business
    """
    if self.in_school:
      return False
    if self.business_id:
      return False

    # Make sure we have sufficient wealth to run a business for a bit
    if self.wealth < MIN_BUSINESS_EXPENSE * 3:
      return False

    # Make sure business is one of our best skills
    total_skill = 0
    for skill_val in self.skills.values():
      total_skill += skill_val
    avg_skill = total_skill / len(self.skills)

    if self.skills["business"] <= avg_skill:
      return False
    print(f"sufficient skill to start business")
    return True

def test_short():
  simulation = EconSimulator(50, 100)
  simulation.run()

  simulation.history.graph()

def test_medium():
  simulation = EconSimulator(200, 100)
  simulation.run()

  simulation.history.graph()

def test_long():
  simulation = EconSimulator(600, 100)
  simulation.run()

  simulation.history.graph()

def test_extra_long():
  simulation = EconSimulator(2000, 100)
  simulation.run()

  simulation.history.graph()

def main():
  test_extra_long()

main()