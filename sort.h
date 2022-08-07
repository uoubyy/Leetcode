#include <vector>

using namespace std;

void quick_sort(vector<int>& nums, int left, int right) {
	if (left + 1 >= right)
		return;

	int l = left, r = right - 1, key = nums[l];

	while (l < r)
	{
		while (l < r && nums[r] >= key)
			r--;
		nums[l] = nums[r];
		while (l < r && nums[l] <= key)
			++l;
		nums[r] = nums[l];
	}
	nums[l] = key;
	quick_sort(nums, left, l);
	quick_sort(nums, l + 1, right);
}

void merge_sort(vector<int>& nums, int left, int right, vector<int>& tmp) {
	if (left + 1 >= right)
		return;

	int mid = left + (right - left) / 2;
	merge_sort(nums, left, mid, tmp);
	merge_sort(nums, mid, right, tmp);

	int i = left, j = mid, k = left;
	while (i < mid || j < right)
	{
		if (j >= right || (i < mid && nums[i] < nums[j]))
			tmp[k++] = nums[i++];
		else
			tmp[k++] = nums[j++];
	}

	for (int i = left; i < right; ++i)
		nums[i] = tmp[i];
}

void insert_sort(vector<int>& nums, int n) {
	for (int i = 0; i < n; ++i)
	{
		for (int j = i; j > 0 && nums[j] > nums[j - 1]; --j)
		{ swap(nums[j], nums[j - 1]); }
	}
}

void shell_sort(vector<int>& nums, int n) {
	for (int gap = n / 2; gap > 0; gap /= 2)
	{
		for (int i = gap; i < n; ++i)
		{
			int temp = nums[i];
			int j;
			for (j = i; j >= gap && nums[j - gap] > temp; j -= gap)
				nums[j] = nums[j - gap];
			nums[j] = temp;
		}
	}
}

void selection_sort(vector<int>& nums, int n) {
	int mid;
	for (int i = 0; i < n - 1; ++i)
	{
		mid = i;
		for (int j = i + 1; j < n; ++j)
		{
			if (nums[j] < nums[mid])
				mid = j;
		}
		swap(nums[mid], nums[i]);
	}
}