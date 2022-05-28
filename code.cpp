#include "env.h"

int backtracing(int n, map<int, int>& steps) {
	if (n <= 0)
		return 0;
	if (steps.count(n))
		return steps[n];

	int num = backtracing(n - 1, steps);
	num += backtracing(n - 2, steps);
	steps[n] = num;
	return num;
}

int climbStairs(int n) {
	map<int, int> steps;
	steps[1] = 1;
	steps[2] = 2;

	return backtracing(n, steps);
}

int getMiniRounds(int n, map<int, int>& steps) {
	if (n < 2)
		return -1;
	if (steps.count(n))
		return steps[n];

	int ans = n;
	if (getMiniRounds(n - 2, steps) > 0)
		ans = min(ans, getMiniRounds(n - 2, steps) + 1);
	if (getMiniRounds(n - 3, steps) > 0)
		ans = min(ans, getMiniRounds(n - 3, steps) + 1);

	ans = (ans == n ? -1 : ans);
	steps[n] = ans;

	return ans;
}

int minimumRounds(vector<int> tasks) {
	map<int, int> taskInfo;
	for (auto task : tasks)
	{
		if (taskInfo.count(task))
			taskInfo[task]++;
		else
			taskInfo[task] = 1;
	}

	int cnt = 0;

	map<int, int> steps;
	steps[2] = 1;
	steps[3] = 1;

	for (auto task : taskInfo)
	{
		if (task.second == 1)
			return -1;

		if (getMiniRounds(task.second, steps) <= 0)
			return -1;

		cnt += getMiniRounds(task.second, steps);
	}

	return cnt;
}

vector<string> readBinaryWatch(int turnedOn) {
	map<int, vector<string>> times;
	for (unsigned int h = 0; h < 12; ++h)
	{
		for (unsigned int m = 0; m < 60; ++m)
		{
			bitset<4> hours(h);
			bitset<6> minutes(m);

			char buffer[10];
			sprintf(buffer, "%d:%02d", hours, minutes);
			times[hours.count() + minutes.count()].push_back(buffer);
		}
	}
	return times[turnedOn];
}

int maxInChild(TreeNode* node, int& curMax) {
	if (node == nullptr)
		return INT_MIN;

	int nodeMax = node->val;

	if (node->left == nullptr && node->right == nullptr)
		return node->val;

	int l = maxInChild(node->left, curMax);
	int r = maxInChild(node->right, curMax);

	bool l_valid = node->left && l > INT_MIN;
	bool r_valid = node->right && r > INT_MIN;

	if (l_valid && r_valid)
		nodeMax = max(
			l,
			max(r, max(node->val, max(node->val + l,
									  max(node->val + r, node->val + l + r)))));
	else if (l_valid)
		nodeMax = max(l, max(node->val, node->val + l));
	else
		nodeMax = max(r, max(node->val, node->val + r));

	curMax = max(curMax, nodeMax);

	if (l_valid && r_valid)
		return max(l + node->val, max(r + node->val, node->val));
	else if (l_valid)
		return max(l + node->val, node->val);
	else
		return max(r + node->val, node->val);

	return node->val;
}

int maxPathSum(TreeNode* root) {
	int curMax = root->val;
	int value = maxInChild(root, curMax);
	return max(curMax, value);
}

int getNext(int i, int n) {
	return (i + 1) % n;
}

int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
	int l = gas.size() - 1, r = 0, sum = gas[l] - cost[l];
	while (l != r)
	{
		int i = sum < 0 ? --l : r++;
		sum += gas[i] - cost[i];
	}
	return sum >= 0 ? l : -1;
}

string largestNumber(vector<int>& nums) {
	vector<vector<string>> cnt(10);

	for (int i = 0; i < nums.size(); ++i)
	{
		int num = nums[i];
		while (num >= 10)
			num /= 10;

		cnt[num].push_back(to_string(nums[i]));
	}

	string ans = "";
	for (int i = 9; i >= 0; --i)
	{
		if (cnt[i].size() > 0)
		{
			sort(cnt[i].begin(), cnt[i].end(),
				 [](string a, string b) { return (a + b) > (b + a); });

			for (auto& str : cnt[i])
				ans += str;
		}
	}

	if (ans[0] == '0')
		return "0";

	return ans;
}

int minMeetingRooms(vector<vector<int>>& intervals) {
	sort(intervals.begin(), intervals.end(),
		 [](vector<int>& a, vector<int>& b) { return a[0] < b[0]; });

	vector<int> rooms;
	rooms.push_back(intervals[0][1]);
	for (int i = 1; i < intervals.size(); ++i)
	{
		bool find = false;
		for (int j = 0; j < rooms.size(); ++j)
		{
			if (intervals[i][0] >= rooms[j])
			{
				rooms[j] = intervals[i][1];
				find = true;
				break;
			}
		}

		if (!find)
			rooms.push_back(intervals[i][1]);
	}

	return rooms.size();
}

bool knows(int a, int b) {
	return false;
}

int findCelebrity(int n) {
	int l = 0, r = 1;
	int host = l;
	while (l < n && r < n)
	{
		if (knows(l, r))
		{
			host = r;
			l = r;
			r++;
		}
		else
		{
			host = l;
			r++;
		}
	}

	for (int i = 0; i < host; ++i)
	{
		if (!knows(i, host))
			return -1;
		if (knows(host, i))
			return -1;
	}

	for (int i = host + 1; i < n; ++i)
	{
		if (!knows(i, host))
			return -1;
		if (knows(host, i))
			return -1;
	}

	return host;
}

// 997
int findJudge(int n, vector<vector<int>>& trust) {
	vector<int> cnt(n + 1, 0);

	for (auto& info : trust)
	{
		cnt[info[0]] = -1;
		if (cnt[info[1]] >= 0)
			cnt[info[1]]++;
	}

	int ans = -1;
	for (int i = 1; i <= n; ++i)
	{
		if (cnt[i] == n - 1)
			ans = i;
	}

	return ans;
}

// 280
void wiggleSort(vector<int>& nums) {
	bool lessEqual = true;

	for (int i = 0; i < nums.size() - 1; ++i)
	{
		if (lessEqual && nums[i] > nums[i + 1])
			swap(nums[i], nums[i + 1]);

		if (!lessEqual && nums[i] < nums[i + 1])
			swap(nums[i], nums[i + 1]);
		lessEqual = !lessEqual;
	}
}

int main() {
	vector<string> ans = readBinaryWatch(2);
	// print_vector(ans);
	return 0;
}