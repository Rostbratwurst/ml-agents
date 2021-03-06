using NUnit.Framework;
using UnityEngine;
using MLAgents;
using MLAgents.Policies;

namespace MLAgents.Tests
{
    [TestFixture]
    public class BehaviorParameterTests
    {
        static void DummyHeuristic(float[] actionsOut)
        {
            // No-op
        }

        [Test]
        public void TestNoModelInferenceOnlyThrows()
        {
            var gameObj = new GameObject();
            var bp = gameObj.AddComponent<BehaviorParameters>();
            bp.behaviorType = BehaviorType.InferenceOnly;

            Assert.Throws<UnityAgentsException>(() =>
            {
                bp.GeneratePolicy(DummyHeuristic);
            });
        }
    }
}
