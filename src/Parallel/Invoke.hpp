// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include "Parallel/Local.hpp"
#include "Parallel/MaxInlineMethodsReached.hpp"
#include "Parallel/TypeTraits.hpp"

namespace Parallel {
/*!
 * \ingroup ParallelGroup
 * \brief Send the data `args...` to the algorithm running on `proxy`, and tag
 * the message with the identifier `temporal_id`.
 *
 * If the algorithm was previously disabled, set `enable_if_disabled` to true to
 * enable the algorithm on the parallel component.
 */
template <typename ReceiveTag, typename Proxy, typename ReceiveDataType>
void receive_data(Proxy&& proxy, typename ReceiveTag::temporal_id temporal_id,
                  ReceiveDataType&& receive_data,
                  const bool enable_if_disabled = false) {
  // Both branches of this if statement call into the charm proxy system, but
  // because we specify [inline] for array chares, we must specify the
  // `ReceiveDataType` explicitly in the template arguments when dispatching to
  // array chares.
  // This is required because of inconsistent overloads in the generated charm++
  // files when [inline] is specified vs when it is not. The [inline] version
  // generates a function signature with template parameters associated with
  // each function argument, ensuring a redundant template parameter for the
  // `ReceiveDataType`. Then, that template parameter cannot be inferred so must
  // be explicitly specified.
  if constexpr (is_array_proxy<std::decay_t<Proxy>>::value) {
    proxy.template receive_data<ReceiveTag, std::decay_t<ReceiveDataType>>(
        std::move(temporal_id), std::forward<ReceiveDataType>(receive_data),
        enable_if_disabled);
  } else {
    proxy.template receive_data<ReceiveTag>(
        std::move(temporal_id), std::forward<ReceiveDataType>(receive_data),
        enable_if_disabled);
  }
}

/*!
 * \ingroup ParallelGroup
 * \brief Send a pointer `message` to the algorithm running on `proxy`.
 *
 * Here, `message` should hold all the information you need as member variables
 * of the object. This includes, temporal ID identifiers, the data itself, and
 * any auxilary information that needs to be communicated. The `ReceiveTag`
 * associated with this `message` should be able unpack the information that was
 * sent.
 *
 * If the component associated with the `proxy` you are calling this on is
 * running on the same charm-node, the exact pointer `message` is sent to the
 * receiving component. No copies of data are done. If the receiving component
 * is on a different charm-node, then the data pointed to by `message` is
 * copied, sent through charm, and unpacked on the receiving component. The
 * pointer that is passed to the algorithm on the receiving component then
 * points to the copied data on the receiving component.
 *
 * \warning You cannot use the `message` pointer after you call this function.
 * Doing so will result in undefined behavior because something else may be
 * controlling the pointer.
 */
template <typename ReceiveTag, typename Proxy, typename MessageType>
void receive_data(Proxy&& proxy, MessageType* message) {
  static_assert(is_array_proxy<std::decay_t<Proxy>>::value or
                    is_array_element_proxy<std::decay_t<Proxy>>::value,
                "Charm++ messages can only be used with Array[Element] chares "
                "at the moment. If you want to use them with other types of "
                "components, you will need to implement it.");
  proxy.template receive_data<ReceiveTag, std::decay_t<MessageType>>(message);
}

/// @{
/*!
 * \ingroup ParallelGroup
 * \brief Invoke a simple action on `proxy`
 */
template <typename Action, typename Proxy>
void simple_action(Proxy&& proxy) {
  proxy.template simple_action<Action>();
}

template <typename Action, typename Proxy, typename Arg0, typename... Args>
void simple_action(Proxy&& proxy, Arg0&& arg0, Args&&... args) {
  proxy.template simple_action<Action, std::decay_t<Arg0>,
                               std::decay_t<Args>...>(
      std::tuple<std::decay_t<Arg0>, std::decay_t<Args>...>(
          std::forward<Arg0>(arg0), std::forward<Args>(args)...));
}
/// @}

/*!
 * \ingroup ParallelGroup
 * \brief Invoke a local synchronous action on `proxy`
 */
template <typename Action, typename Proxy, typename... Args>
decltype(auto) local_synchronous_action(Proxy&& proxy, Args&&... args) {
  return Parallel::local_branch(proxy)
      ->template local_synchronous_action<Action>(std::forward<Args>(args)...);
}

/// @{
/*!
 * \ingroup ParallelGroup
 * \brief Invoke a threaded action on `proxy`, where the proxy must be a
 * nodegroup.
 */
template <typename Action, typename Proxy>
void threaded_action(Proxy&& proxy) {
  proxy.template threaded_action<Action>();
}

template <typename Action, typename Proxy, typename Arg0, typename... Args>
void threaded_action(Proxy&& proxy, Arg0&& arg0, Args&&... args) {
  proxy.template threaded_action<Action>(
      std::tuple<std::decay_t<Arg0>, std::decay_t<Args>...>(
          std::forward<Arg0>(arg0), std::forward<Args>(args)...));
}
/// @}
}  // namespace Parallel
