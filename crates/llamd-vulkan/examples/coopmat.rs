fn main() {
    match llamd_vulkan::coopmat_report() {
        Some(r) if !r.is_empty() => print!("{r}"),
        Some(_) => println!("extension present, zero shapes"),
        None => println!("no VK_KHR_cooperative_matrix"),
    }
}
