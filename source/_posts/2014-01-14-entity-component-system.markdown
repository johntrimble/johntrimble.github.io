---
layout: post
title: "Entity Component System"
date: 2014-01-14 11:55:23 -0700
comments: true
categories: 
---
About a year ago, I was listening to a [talk](http://www.youtube.com/watch?v=V1Eu9vZaDYw) by Chris Granger on how he leveraged the Entity Component System (ECS) architectural pattern when building LightTable. I'd never heard of the pattern before, probably because it typically finds its use in video games, and I don't really do anything in the way of video game development. Chris did a pretty good job selling the approach, so I decided to try it for myself. A couple of months ago, I started working on an Asteroids-like [game](https://github.com/johntrimble/asteroids) as a side project to evaluate this pattern, my particular interest being its effectiveness at facilitating code reuse, decoupling, and testing. It's not done, but I'm sufficiently far enough along to get a sense of how the pattern plays out in practice. I won't go over what the ECS is (check out Chris's talk or the Wikipedia [article](http://en.wikipedia.org/wiki/Entity_component_system) if you're interested in that), but instead just give a brief evaluation of its effectiveness.

## Code Reuse
This does turn out to be a real win. By simply adding a new property to an entity, I can endow it with new behavior. For example, by giving the camera entity a `movement` component, I can make it so that it rotates or scrolls. I can also make the ship explode in the same manner as the asteroids by giving it an `asteroid-explosive-death` component. This means that once I add a new set of components and systems, I can take advantage of them on any entity that I'd like. Definitely a big win for code reuse!

## Decoupling
The story here is a bit more mixed. On the one hand, the different systems do not directly call each other, and this leads some people to conclude that they aren't coupled, but lets take a look at that. Here's the function that takes a world at time t<sub>1</sub> and generates a world at time t<sub>2</sub>:

```clojure
(defn next-world [world]
  (-> world
      keyboard/keyboard-system
      intents/intent-system
      intents/rotation-system
      intents/thrust-system
      projectile/firing-system
      physics/physics-system
      physics/collision-detection-system
      projectile/projectile-collision-resolution-system
      physics/collision-physics-system
      health/impulse-damage-system
      health/damage-resolution-system
      asteroids/asteroid-death-system
      health/health-bar-system
      core/ttl-system))
```

In general, it looks pretty simple. Each system takes a world state as a parameter and produces a world state as its value. Then the ability to move a game state forward is realized by composing the various systems together. It's in that composing bit where the complexity hides: the order in which the systems are composed together matters! For example, the `impulse-damage-system` must be applied after the `collision-physics-system` as the former is looking for entities with an `impulse` component which is added by the `collision-physics-system`. One cannot simply look at the systems in isolation, but must have an awareness of how the systems interact with each other, which increases the cognitive load when adding, modifying, or removing systems. On the plus side, the systems are at least loosely coupled as they just care about the world being in some state, not about who put it in that state.

Of course, one could argue that this complexity is inherent in the problem (i.e. there's just no avoiding it), or not a failing of entity component system, but rather an issue with the way I applied that pattern. The important thing here is to keep in mind that the lack of a direct function call doesn't necessarily mean two parts of a system are decoupled, and it's valuable to identify these areas of complexity because they help you understand where the problems are going to arise.

## Testability
This turned out to be a huge win. Since none of the systems call each other, each system only cares about the world state passed into it, and none of them (at least none of the ones listed above) have any side effects, testing is a breeze! Here's what one of my unit tests for the `physics-system` looks like:

```clojure
(let [a (core/entity (core/movement [1 0.5]
                                    [2 1]
                                    math/infinity
                                    0
                                    0
                                    math/infinity)
                     (core/position [5 5])
                     (core/aabb [4 4] [6 6]))
      world (core/assoc-entity {} a)
      a (core/get-entity world (core/get-id a))
      world-new (physics/physics-system world)
      a-new (core/get-entity world-new (core/get-id a))]
  (describe "physics-system"
            (it "should update velocity by acceleration"
                (should= [3 1.5] (core/get-velocity a-new)))
            (it "should update velocity before updating position"
                (should-not= [7 6] (core/get-position a-new)))
            (it "should update the position"
                (should= [8 6.5] (core/get-position a-new)))
            (it "should provide a valid value for rotation"
                (should-not (js/isNaN (core/get-rotation a-new))))
            (it "should provide a valid value for angular components"
                (should-not (js/isNaN (core/get-angular-velocity a-new)))
                (should-not (js/isNaN (core/get-angular-acceleration a-new))))
            (it "should update the aabb"
                (should= [[7 5.5] [9 7.5]] (core/get-aabb a)))))
```

Notice how there aren't any mocks. I just build a world state, call the system, and verify that the new world state looks the way it should. I wish that everything I did was this straight forward to test!

## Conclusion
In general, I found that entity component system, as an architectural pattern, really shined in regards to code reuse and testability. I did not find the different systems as insulated from each other as I originally expected, but nonetheless, the complexity is manageable, especially with how easy testing is. My next step--other than actually finishing the game--is to investigate its viability in contexts outside of game development. LightTable just recently became open source, and since Chris Granger's talk on how he used this pattern in LightTable got me interested in the first place, I'll probably start there by analyzing its source code.
