#include <vector>

using namespace std;

void quick_sort(vector<int>& nums, int left, int right)
{
    if (left + 1 >= right)
        return;

    int l = left, r = right - 1, key = nums[l];

    while (l < r) {
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

void merge_sort(vector<int>& nums, int left, int right, vector<int>& tmp)
{
    if (left + 1 >= right)
        return;

    int mid = left + (right - left) / 2;
    merge_sort(nums, left, mid, tmp);
    merge_sort(nums, mid, right, tmp);

    int i = left, j = mid, k = left;
    while (i < mid || j < right) {
        if (j >= right || (i < mid && nums[i] < nums[j]))
            tmp[k++] = nums[i++];
        else
            tmp[k++] = nums[j++];
    }

    for (int i = left; i < right; ++i)
        nums[i] = tmp[i];
}